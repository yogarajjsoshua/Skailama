from langgraph.graph import StateGraph, END
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

    # terminal edges for short-circuit nodes
    graph.add_edge("unsupported", END)
    graph.add_edge("clarification", END)

    # happy-path pipeline
    graph.add_edge("trigger_detection", "state_assembly")
    graph.add_edge("state_assembly", "validation")
    graph.add_edge("validation", END)

    return graph.compile()


mini_promotion_graph = build_graph()

