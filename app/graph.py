from langgraph.graph import StateGraph, END
from app.state import PromotionState
from app.nodes import (
    intent_classification_node,
    trigger_detection_node,
    state_assembly_node,
    validation_node,
)


def build_graph():
    graph = StateGraph(PromotionState)

    graph.add_node("intent_classification", intent_classification_node)
    graph.add_node("trigger_detection", trigger_detection_node)
    graph.add_node("state_assembly", state_assembly_node)
    graph.add_node("validation", validation_node)

    graph.set_entry_point("intent_classification")
    graph.add_edge("intent_classification", "trigger_detection")
    graph.add_edge("trigger_detection", "state_assembly")
    graph.add_edge("state_assembly", "validation")
    graph.add_edge("validation", END)

    return graph.compile()


mini_promotion_graph = build_graph()
