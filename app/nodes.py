import json
from langsmith import traceable
from app.llm import llm
from app.pormpts import (
    INTENT_CLASSIFICATION_PROMPT,
    TRIGGER_ONLY_CLASSIFICATION_PROMPT,
    VALIDATION_CLASSIFICATION_PROMPT,
)
from app.state import PromotionState


@traceable(name="intent_classification_node", run_type="chain")
def intent_classification_node(state: PromotionState) -> PromotionState:
    raw = llm(INTENT_CLASSIFICATION_PROMPT, state["message"])
    result = json.loads(raw)
    state["feature"] = result.get("intent", "unsupported")
    return state


@traceable(name="trigger_detection_node", run_type="chain")
def trigger_detection_node(state: PromotionState) -> PromotionState:
    prompt = TRIGGER_ONLY_CLASSIFICATION_PROMPT + f"\nDetected intent: {state['feature']}"
    raw = llm(TRIGGER_ONLY_CLASSIFICATION_PROMPT, state["message"])
    result = json.loads(raw)
    state["tiers"] = result.get("tiers", [])
    return state


@traceable(name="state_assembly_node", run_type="chain")
def state_assembly_node(state: PromotionState) -> PromotionState:
    state["tier_behavior"] = "best_tier_only"
    state["customer_eligibility"] = []
    state["status"] = "pending_validation"
    state["blockers"] = []
    return state


@traceable(name="validation_node", run_type="chain")
def validation_node(state: PromotionState) -> PromotionState:
    payload = json.dumps([{"id": 1, "text": state["message"], "label": state["feature"]}])
    raw = llm(VALIDATION_CLASSIFICATION_PROMPT, payload)
    result = json.loads(raw)

    items = result.get("results", [])
    if items and items[0].get("is_correct") is False:
        state["status"] = "unsupported"
        state["blockers"] = [{"field": "intent", "reason": items[0].get("reason", "")}]
    else:
        state["status"] = "supported"
        state["blockers"] = []

    return state