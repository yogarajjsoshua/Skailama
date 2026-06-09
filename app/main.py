import json
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import AzureOpenAI, APIConnectionError, AuthenticationError
from langsmith import traceable, Client as LangSmithClient
from langgraph.types import Command
import os
import logging
from dotenv import load_dotenv

load_dotenv()  # must run before langsmith reads LANGCHAIN_* vars

from datetime import datetime
import app.pormpts as prompts
from app.graph import mini_promotion_graph
from app.db.mongo import MongoDBClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db_client = MongoDBClient()

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_4_KEY"),
    api_version=os.getenv("OPENAI_API_4_VERSION"),
    azure_endpoint=os.getenv("OPENAI_4_BASE_URL"),
)

DEPLOYMENT_NAME = os.getenv("OPEN_API_4_ENGINE")


def check_azure_openai_connection() -> None:
    logger.info("Checking Azure OpenAI connection")
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        model_used = response.model
        logger.info(
            "OpenAI connection successful! "
            f"Deployment: '{DEPLOYMENT_NAME}', Model: '{model_used}'"
        )
    except AuthenticationError as e:
        logger.error(f"authentication failed: {e}")
        raise RuntimeError("authentication failed") from e
    except APIConnectionError as e:
        logger.error(f"connection error: {e}")
        raise RuntimeError(" your base URL eRROR.") from e
    except Exception as e:
        logger.error(f"connectivity check failed: {e}")
        raise RuntimeError(f"connectivity check failed: {e}") from e


def check_langsmith_connection() -> None:
    """Verify LangSmith API key and connectivity at startup.

    Attempts to reach the LangSmith API and list projects.
    Logs a clear message for each failure mode so you know exactly
    what went wrong (missing key, bad key, or network issue).
    Does NOT raise — a tracing outage should not kill the server.
    """
    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    project = os.getenv("LANGCHAIN_PROJECT", "default")

    if not tracing_enabled:
        logger.warning("LangSmith tracing is DISABLED (LANGCHAIN_TRACING_V2 != 'true')")
        return

    if not api_key or api_key == "YOUR_LANGSMITH_API_KEY_HERE":
        logger.warning(
            "LangSmith API key is not set. "
            "Add your key to .env: LANGCHAIN_API_KEY=ls__..."
        )
        return

    try:
        ls_client = LangSmithClient(
            api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
            api_key=api_key,
        )
        # list_projects() makes a real HTTP call — cheapest connectivity check
        projects = list(ls_client.list_projects())
        project_names = [p.name for p in projects]
        if project not in project_names:
            logger.info(
                f"LangSmith connected ✓  "
                f"(project '{project}' will be auto-created on first trace)"
            )
        else:
            logger.info(
                f"LangSmith connected ✓  "
                f"Tracing to project: '{project}'"
            )
    except Exception as e:
        err = str(e)
        if "401" in err or "Unauthorized" in err or "403" in err:
            logger.error(
                "LangSmith connection FAILED — invalid API key. "
                "Check LANGCHAIN_API_KEY in your .env file."
            )
        elif "ConnectionError" in err or "ConnectTimeout" in err or "Name or service" in err:
            logger.error(
                "LangSmith connection FAILED — network error. "
                f"Could not reach {os.getenv('LANGCHAIN_ENDPOINT', 'https://api.smith.langchain.com')}."
            )
        else:
            logger.error(f"LangSmith connection FAILED — {err}")


def check_mongodb_connection() -> None:
    logger.info("Checking MongoDB connection")
    try:
        db_client.client.admin.command('ping')
        logger.info("MongoDB connected ✓")
    except Exception as e:
        logger.error(f"MongoDB connection FAILED — {e}")
        raise RuntimeError(f"MongoDB connection FAILED — {e}") from e


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    check_azure_openai_connection()
    check_langsmith_connection()
    check_mongodb_connection()
    yield
    # --- Shutdown (add cleanup here if needed) ---


app = FastAPI(lifespan=lifespan)




@app.get("/")
def read_root():
    return {"message": "Hello World"}


class ChatRequest(BaseModel):
    message: str
    # thread_id is now SERVER-OWNED — do NOT supply it for new conversations.
    # For clarify calls, use ClarifyRequest which requires thread_id.


class ClarifyRequest(BaseModel):
    thread_id: str
    clarification: str


@app.post("/chat")
def parsePromotionRequest(request: ChatRequest):
    logger.info(f"request: {request}")
    message = request.message
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": message}],
        max_tokens=1,
    )
    reply = response.choices[0].message.content
    return {"reply": reply}

@app.post("/chat/intent-classifier/llm")
@traceable(name="intent-classifier", run_type="chain")
def intent_classifier(request: ChatRequest):
    logger.info(f"request: {request}")
    message = request.message
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": prompts.INTENT_CLASSIFICATION_PROMPT},
            {"role": "user", "content": message}],
        response_format={"type": "json_object"},
        temperature=0
    )
    reply = response.choices[0].message.content
    reply_dict = json.loads(reply)
    return {"reply": reply_dict["intent"]}


@app.post("/chat/intent-classifier/triggers/llm")
@traceable(name="intent-triggers-classifier", run_type="chain")
def intent_triggers_classifier(request: ChatRequest):
    logger.info(f"request: {request}")
    message = request.message
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": prompts.INTENT_CLASSIFICATION_FEATURE_TRIGGER_PROMPT},
            {"role": "user", "content": message}],
        response_format={"type": "json_object"},
        temperature=0
    )
    reply = response.choices[0].message.content
    reply_dict = json.loads(reply)
    return {"reply": reply_dict}



def _to_dict(state) -> dict:
    if hasattr(state, "model_dump"):
        return state.model_dump()
    elif hasattr(state, "dict"):
        return state.dict()
    return dict(state)

def _save_graph_result_to_db(thread_id: str, result, is_interrupt: bool = False, interrupted_value: dict = None):
    try:
        state_dict = _to_dict(result)
        db_client.save_chat_context(user_id="default_user", chat_id=thread_id, context=state_dict)

        if is_interrupt and interrupted_value:
            db_client.save_message(
                user_id="default_user",
                chat_id=thread_id,
                message_id=str(uuid.uuid4()),
                role="assistant",
                content=interrupted_value.get("question", ""),
                timestamp=datetime.now().isoformat(),
                ui={"status": "clarification", "blockers": interrupted_value.get("blockers", [])}
            )
        else:
            final_reply = {
                "feature": state_dict.get("feature"),
                "status": state_dict.get("status"),
                "blockers": state_dict.get("blockers", []),
                "missing_fields": state_dict.get("missing_fields", []),
                "tiers": state_dict.get("tiers", []),
                "tier_behavior": state_dict.get("tier_behavior"),
                "customer_eligibility": state_dict.get("customer_eligibility", []),
            }
            db_client.save_message(
                user_id="default_user",
                chat_id=thread_id,
                message_id=str(uuid.uuid4()),
                role="assistant",
                content=f"Promotion feature: {final_reply.get('feature')}, status: {final_reply.get('status')}",
                timestamp=datetime.now().isoformat(),
                ui=final_reply,
            )
    except Exception as e:
        logger.warning("Failed to save graph result to MongoDB: %s", e)


@app.post("/chat/mini-promotion-agent")
@traceable(name="mini-promotion-agent", run_type="chain")
def mini_promotion_agent(request: ChatRequest):
    # Server creates and owns thread_id — returned to client in the response
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        db_client.create_chat_if_missing(user_id="default_user", chat_id=thread_id)
        db_client.save_message(
            user_id="default_user",
            chat_id=thread_id,
            message_id=str(uuid.uuid4()),
            role="user",
            content=request.message,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.warning("Failed to save initial message to MongoDB: %s", e)

    initial_state = {
        "message": request.message,
        "history": [],
        "clarification_attempts": 0,
        "thread_id": thread_id,
    }
    result = mini_promotion_graph.invoke(initial_state, config=config)

    # Detect if the graph paused at a clarification interrupt
    import time
    for i in range(10):
        state_snapshot = mini_promotion_graph.get_state(config)
        logger.info(f"[Retry {i}] next={state_snapshot.next} tasks_len={len(state_snapshot.tasks) if state_snapshot.tasks else 0}")
        if state_snapshot.next:
            if state_snapshot.tasks and state_snapshot.tasks[0].interrupts:
                logger.info(f"[Retry {i}] Interrupt found!")
                interrupted_value = state_snapshot.tasks[0].interrupts[0].value
                
                # Save status to db
                _save_graph_result_to_db(thread_id, state_snapshot.values, is_interrupt=True, interrupted_value=interrupted_value)
                
                return {
                    "reply": {
                        **interrupted_value,           # spreads status/blockers/question from interrupt()
                        "thread_id": thread_id,
                    }
                }
            else:
                logger.info(f"[Retry {i}] Next exists but no interrupts yet.")
            time.sleep(0.1)
        else:
            logger.info(f"[Retry {i}] next is empty, graph finished.")
            break

    # Append the final structured bot response to history as a 'system' turn
    final_reply = {
        "feature": result.get("feature"),
        "status": result.get("status"),
        "blockers": result.get("blockers", []),
        "missing_fields": result.get("missing_fields", []),
        "tiers": result.get("tiers", []),
        "tier_behavior": result.get("tier_behavior"),
        "customer_eligibility": result.get("customer_eligibility", []),
    }
    _append_bot_reply_to_history(config, result, json.dumps(final_reply))

    _save_graph_result_to_db(thread_id, result)

    return {"reply": {**final_reply, "thread_id": thread_id}}


@app.post("/chat/mini-promotion-agent/clarify")
@traceable(name="mini-promotion-agent-clarify", run_type="chain")
def mini_promotion_agent_clarify(request: ClarifyRequest):
    """Resume a paused clarification graph with the user's clarified message."""
    config = {"configurable": {"thread_id": request.thread_id}}

    # Verify there is actually a paused graph for this thread
    state_snapshot = mini_promotion_graph.get_state(config)
    if not state_snapshot.next:
        raise HTTPException(
            status_code=400,
            detail=f"No paused clarification found for thread_id '{request.thread_id}'. "
                   "Start a new conversation via POST /chat/mini-promotion-agent."
        )

    try:
        db_client.save_message(
            user_id="default_user",
            chat_id=request.thread_id,
            message_id=str(uuid.uuid4()),
            role="user",
            content=request.clarification,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.warning("Failed to save clarification message to MongoDB: %s", e)

    # Resume the graph — Command(resume=...) passes the user's text back to interrupt()
    result = mini_promotion_graph.invoke(
        Command(resume=request.clarification),
        config=config,
    )

    # Check if the graph paused again (another round of clarification)
    import time
    for i in range(10):
        state_snapshot = mini_promotion_graph.get_state(config)
        logger.info(f"[Clarify Retry {i}] next={state_snapshot.next} tasks_len={len(state_snapshot.tasks) if state_snapshot.tasks else 0}")
        if state_snapshot.next:
            if state_snapshot.tasks and state_snapshot.tasks[0].interrupts:
                logger.info(f"[Clarify Retry {i}] Interrupt found!")
                interrupted_value = state_snapshot.tasks[0].interrupts[0].value
                
                _save_graph_result_to_db(request.thread_id, state_snapshot.values, is_interrupt=True, interrupted_value=interrupted_value)
                
                return {
                    "reply": {
                        **interrupted_value,
                        "thread_id": request.thread_id,
                    }
                }
            else:
                logger.info(f"[Clarify Retry {i}] Next exists but no interrupts yet.")
            time.sleep(0.1)
        else:
            logger.info(f"[Clarify Retry {i}] next is empty, graph finished.")
            break

    # Append the final structured bot response to history as a 'system' turn
    final_reply = {
        "feature": result.get("feature"),
        "status": result.get("status"),
        "blockers": result.get("blockers", []),
        "missing_fields": result.get("missing_fields", []),
        "tiers": result.get("tiers", []),
        "tier_behavior": result.get("tier_behavior"),
        "customer_eligibility": result.get("customer_eligibility", []),
    }
    _append_bot_reply_to_history(config, result, json.dumps(final_reply))

    _save_graph_result_to_db(request.thread_id, result)

    return {"reply": {**final_reply, "thread_id": request.thread_id}}


# ---------------------------------------------------------------------------
# History helper
# ---------------------------------------------------------------------------

def _append_bot_reply_to_history(config: dict, result: dict, reply_json: str) -> None:
    """Update the checkpoint state to append the bot's final reply as a 'system' turn."""
    try:
        current = mini_promotion_graph.get_state(config)
        updated_history = (current.values.get("history") or []) + [
            {"role": "system", "content": f"[Bot] Final response: {reply_json}"}
        ]
        mini_promotion_graph.update_state(
            config,
            {"history": updated_history},
        )
    except Exception as e:
        logger.warning("Failed to append bot reply to history: %s", e)


# ---------------------------------------------------------------------------
# History endpoint
# ---------------------------------------------------------------------------

@app.get("/chat/mini-promotion-agent/{thread_id}/history")
def get_conversation_history(thread_id: str):
    """Return the accumulated conversation history for a given thread.

    Each entry has:
      - role : 'user'   — a message sent by the human
      - role : 'system' — a message sent by the chatbot (clarification question or final reply)
    """
    try:
        messages = db_client.get_messages(user_id="default_user", chat_id=thread_id)
        if messages:
            history = []
            for msg in messages:
                role = "system" if msg.get("role") == "assistant" else msg.get("role", "user")
                history.append({
                    "role": role,
                    "content": msg.get("content", "")
                })
            return {"thread_id": thread_id, "history": history}
    except Exception as e:
        logger.warning("Failed to fetch history from MongoDB: %s", e)

    config = {"configurable": {"thread_id": thread_id}}
    state_snapshot = mini_promotion_graph.get_state(config)
    if not state_snapshot or not state_snapshot.values:
        raise HTTPException(
            status_code=404,
            detail=f"No conversation found for thread_id '{thread_id}'."
        )
    history = state_snapshot.values.get("history", [])
    return {"thread_id": thread_id, "history": history}
