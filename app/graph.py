from langgraph.graph import StateGraph, END
from app.mongo_checkpointer import MongoEngineCheckpointer
from app.state import PromotionState
from app.nodes import (
    intent_classification_node,
    trigger_detection_node,
    state_assembly_node,
    validation_node,
    unsupported_node,
    clarification_node,
)

SUPPORTED_INTENTS = {"free_gift", "buy_x_get_y", "tiered_discount"}


def _route_after_intent(state: PromotionState) -> str:
    """Conditional router: short-circuit to terminal nodes for unsupported/clarification."""
    feature = state.get("feature", "unsupported")
    if feature == "clarification":
        return "clarification"
    if feature not in SUPPORTED_INTENTS:
        return "unsupported"
    return "trigger_detection"


def build_graph():
    graph = StateGraph(PromotionState)

    # nodes
    graph.add_node("intent_classification", intent_classification_node)
    graph.add_node("unsupported", unsupported_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("trigger_detection", trigger_detection_node)
    graph.add_node("state_assembly", state_assembly_node)
    graph.add_node("validation", validation_node)

    # entry + conditional routing
    graph.set_entry_point("intent_classification")
    graph.add_conditional_edges(
        "intent_classification",
        _route_after_intent,
        {
            "unsupported": "unsupported",
            "clarification": "clarification",
            "trigger_detection": "trigger_detection",
        },
    )

    # clarification loops back to intent_classification after user provides input
    # (interrupt() inside clarification_node pauses execution between these two edges)
    graph.add_edge("clarification", "intent_classification")

    # unsupported is truly terminal
    graph.add_edge("unsupported", END)

    # happy-path pipeline
    graph.add_edge("trigger_detection", "state_assembly")
    graph.add_edge("state_assembly", "validation")
    graph.add_edge("validation", END)

    # MongoEngineCheckpointer is required for interrupt() to persist state across calls
    MongoEngineCheckpointer.connect_db()
    checkpointer = MongoEngineCheckpointer()
    return graph.compile(checkpointer=checkpointer)


mini_promotion_graph = build_graph()
