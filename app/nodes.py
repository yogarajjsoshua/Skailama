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
    # Initialise history list on the very first call
    history = state.get("history") or []

    # Append the current user message to history
    message = state["message"]
    history = history + [{"role": "user", "content": message}]

    # Pass history so the LLM has full conversation context
    raw = llm(INTENT_CLASSIFICATION_PROMPT, message, history=history)
    result = json.loads(raw)

    state["feature"] = result.get("intent", "unsupported")
    state["history"] = history

    # Initialise defaults so all downstream nodes always have valid keys
    state["tiers"] = []
    state["tier_behavior"] = ""
    state["customer_eligibility"] = []
    state["status"] = ""
    state["blockers"] = []
    state["missing_fields"] = []
    return state


@traceable(name="unsupported_node", run_type="chain")
def unsupported_node(state: PromotionState) -> PromotionState:
    """Terminal node for unsupported intents.
    Builds the standard {status, blockers} response structure.
    Appends the bot's rejection message to history as a 'system' turn.
    """
    reason = (
        f"The requested promotion type '{state.get('feature', 'unknown')}' "
        "is not supported by this app."
    )
    state["status"] = "unsupported"
    state["blockers"] = [{"field": "intent", "reason": reason}]

    # Record the bot response in history so future turns have full context
    bot_message = f"[Bot] Unsupported promotion type: {reason}"
    state["history"] = (state.get("history") or []) + [
        {"role": "system", "content": bot_message}
    ]
    return state


@traceable(name="clarification_node", run_type="chain")
def clarification_node(state: PromotionState) -> PromotionState:
    """Human-in-the-loop node for clarification intents.
    Pauses execution via interrupt(), surfaces the question to the API caller,
    then resumes when the user provides their clarification text.

    History tracking
    ----------------
    Before pausing  → append the bot's clarification question as a 'system' turn.
    After resuming  → append the user's clarification reply as a 'user' turn.
    The clarified message then replaces state["message"] and the graph loops back
    to intent_classification for re-classification.
    """
    from langgraph.types import interrupt  # local import to keep module-level clean

    MAX_CLARIFICATION_ATTEMPTS = 3

    attempts = state.get("clarification_attempts", 0)
    history  = state.get("history") or []

    # Safety valve: if we've asked too many times, treat as unsupported
    if attempts >= MAX_CLARIFICATION_ATTEMPTS:
        bot_message = (
            f"[Bot] Unable to determine the promotion intent after "
            f"{MAX_CLARIFICATION_ATTEMPTS} clarification attempts."
        )
        state["status"] = "unsupported"
        state["blockers"] = [{"field": "intent", "reason": bot_message}]
        state["history"] = history + [{"role": "system", "content": bot_message}]
        return state

    # Build the clarification question to surface to the user
    clarification_question = (
        "Could you clarify the promotion? "
        "Describe the trigger and the reward product/discount."
    )
    question_payload = {
        "status": "clarification",
        "blockers": [
            {
                "field": "intent",
                "reason": (
                    "Your request is missing key details. "
                    "Please specify: the trigger condition (e.g. spend $X, buy N items) "
                    "and the exact reward product or discount so we can configure the promotion."
                ),
            }
        ],
        "question": clarification_question,
        "attempt": attempts + 1,
    }

    # Record the bot's clarification question in history BEFORE pausing
    history = history + [{"role": "system", "content": f"[Bot] {clarification_question}"}]
    state["history"] = history

    # Pause here — execution resumes when the API caller sends a clarification
    clarification_text = interrupt(question_payload)

    # Append the user's clarification reply to history AFTER resuming
    state["history"] = history + [{"role": "user", "content": clarification_text}]

    # Update state with the user's clarified message and reset for re-classification
    state["message"] = clarification_text
    state["clarification_attempts"] = attempts + 1
    state["status"] = ""
    state["blockers"] = []
    state["missing_fields"] = []
    return state


@traceable(name="trigger_detection_node", run_type="chain")
def trigger_detection_node(state: PromotionState) -> PromotionState:
    # Pass history so trigger LLM knows the full conversation context
    raw = llm(TRIGGER_ONLY_CLASSIFICATION_PROMPT, state["message"], history=state.get("history"))
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
    raw = llm(VALIDATION_CLASSIFICATION_PROMPT, payload, history=state.get("history"))
    result = json.loads(raw)

    items = result.get("results", [])
    if items and items[0].get("is_correct") is False:
        state["status"] = "unsupported"
        state["blockers"] = [{"field": "intent", "reason": items[0].get("reason", "")}]
    else:
        state["status"] = "supported"
        state["blockers"] = []

    return state