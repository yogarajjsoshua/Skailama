import json
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from openai import AzureOpenAI, APIConnectionError, AuthenticationError
from langsmith import traceable
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    check_azure_openai_connection()
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
