import json
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from openai import AzureOpenAI, APIConnectionError, AuthenticationError
from langsmith import traceable, Client as LangSmithClient
import os
import logging
from dotenv import load_dotenv

load_dotenv()  # must run before langsmith reads LANGCHAIN_* vars

import app.pormpts as prompts
from app.graph import mini_promotion_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    check_azure_openai_connection()
    check_langsmith_connection()
    yield
    # --- Shutdown (add cleanup here if needed) ---


app = FastAPI(lifespan=lifespan)




@app.get("/")
def read_root():
    return {"message": "Hello World"}


class ChatRequest(BaseModel):
    message: str


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



@app.post("/chat/mini-promotion-agent")
@traceable(name="mini-promotion-agent", run_type="chain")
def mini_promotion_agent(request: ChatRequest):
    result = mini_promotion_graph.invoke({"message": request.message})
    return {
        "reply": {
            "feature": result.get("feature"),
            "tiers": result.get("tiers", []),
            "tier_behavior": result.get("tier_behavior"),
            "customer_eligibility": result.get("customer_eligibility", []),
            "status": result.get("status"),
            "blockers": result.get("blockers", []),
        }
    }
