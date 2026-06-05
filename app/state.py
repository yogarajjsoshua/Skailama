from typing import TypedDict, List, Any


class PromotionState(TypedDict):
    message: str
    feature: str
    tiers: List[Any]
    tier_behavior: str
    customer_eligibility: List[Any]
    status: str
    blockers: List[Any]
