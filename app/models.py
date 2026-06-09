from typing import Dict, Any, Optional, List, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


# ---------------------------------------------------------------------------
# Trigger Schema Models
# ---------------------------------------------------------------------------

class TriggerType(str, Enum):
    cart_quantity = "cart_quantity"
    cart_subtotal = "cart_subtotal"
    collection_quantity = "collection_quantity"
    collection_subtotal = "collection_subtotal"
    product_quantity = "product_quantity"
    product_subtotal = "product_subtotal"


class TriggerOperator(str, Enum):
    gte = ">="
    gt = ">"
    lte = "<="
    lt = "<"
    eq = "="


class ScopeType(str, Enum):
    collection = "collection"
    product = "product"


class TriggerScope(BaseModel):
    """Scope restricts the trigger to a specific collection or product set."""

    type: ScopeType = Field(..., description="Scope type: 'collection' or 'product'")
    collectionTitles: Union[List[str], Literal["all"]] = Field(
        ...,
        description=(
            "List of collection/product titles to include, "
            "or the string 'all' to apply to every collection/product."
        ),
    )

    @field_validator("collectionTitles", mode="before")
    @classmethod
    def validate_collection_titles(cls, v):
        if v == "all":
            return v
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("collectionTitles must be a non-empty list or the string 'all'")
            return v
        raise ValueError("collectionTitles must be a list of strings or the string 'all'")


class TriggerModel(BaseModel):
    """Validated trigger object inside a tier."""

    type: TriggerType = Field(..., description="Trigger type (e.g. collection_quantity)")
    operator: TriggerOperator = Field(..., description="Comparison operator (>=, >, <=, <, =)")
    value: Union[int, float] = Field(..., gt=0, description="Threshold value (must be > 0)")
    currency: Optional[str] = Field(
        None,
        description="ISO-4217 currency code (e.g. USD, EUR). Required for subtotal triggers.",
    )
    scope: Optional[TriggerScope] = Field(
        None,
        description=(
            "Optional scope limiting the trigger to specific collections or products. "
            "Required when type is collection_* or product_*."
        ),
    )

    @model_validator(mode="after")
    def check_currency_for_subtotal(self):
        subtotal_types = {
            TriggerType.cart_subtotal,
            TriggerType.collection_subtotal,
            TriggerType.product_subtotal,
        }
        if self.type in subtotal_types and not self.currency:
            raise ValueError(
                f"'currency' is required when trigger type is '{self.type.value}'"
            )
        return self

    @model_validator(mode="after")
    def check_scope_for_scoped_triggers(self):
        scoped_types = {
            TriggerType.collection_quantity,
            TriggerType.collection_subtotal,
            TriggerType.product_quantity,
            TriggerType.product_subtotal,
        }
        if self.type in scoped_types and self.scope is None:
            raise ValueError(
                f"'scope' is required when trigger type is '{self.type.value}'"
            )
        return self


# ---------------------------------------------------------------------------
# Reward Schema Models
# ---------------------------------------------------------------------------

class RewardType(str, Enum):
    # Tiered discount rewards
    percentage_off = "percentage_off"
    fixed_amount_off = "fixed_amount_off"
    # Buy X Get Y rewards
    percentage_off_y = "percentage_off_y"
    fixed_amount_off_y = "fixed_amount_off_y"
    # Free gift rewards
    free_gift = "free_gift"
    free_gift_product = "free_gift_product"


class ResolutionStatus(str, Enum):
    resolved = "resolved"
    admin_selection_required = "admin_selection_required"
    unresolved = "unresolved"


class ProductTarget(BaseModel):
    """Resolution metadata for a product that may or may not be resolved yet."""
    type: Literal["product"] = "product"
    status: ResolutionStatus = Field(..., description="Resolution status of the product")
    query: str = Field(..., description="User-provided product search term")
    resolved_id: Optional[str] = Field(
        None,
        description="Resolved product ID (null when admin_selection_required or unresolved)",
    )


class GiftProductTarget(BaseModel):
    """Resolution metadata specifically for a free gift product."""
    status: ResolutionStatus = Field(..., description="Resolution status of the gift product")
    query: str = Field(..., description="User-provided gift description/search term")
    resolved_id: Optional[str] = Field(
        None,
        description="Resolved product ID (null when admin_selection_required or unresolved)",
    )


class RewardModel(BaseModel):
    """
    Dynamic reward object whose shape depends on `type`.

    Supported types
    ---------------
    percentage_off      – tiered: { type, value }
    fixed_amount_off    – tiered: { type, value }
    percentage_off_y    – buy-X-get-Y: { type, value, y_target, quantity }
                          value=100 represents a completely free Y
    fixed_amount_off_y  – buy-X-get-Y: { type, value, y_target, quantity }
    free_gift           – free gift: { type, gift_product, quantity }
    free_gift_product   – simple gift: { type, value } (legacy / simple form)
    """

    type: RewardType = Field(..., description="Reward type")

    # ------ Shared numeric value (tiered / buy-x-get-y percentage) ------
    value: Optional[Union[int, float]] = Field(
        None,
        description=(
            "Discount magnitude. Required for percentage_off, fixed_amount_off, "
            "percentage_off_y, fixed_amount_off_y, and free_gift_product. "
            "For percentage_off_y with value=100 this represents a free Y item."
        ),
    )

    # ------ Buy X Get Y fields ------
    y_target: Optional[ProductTarget] = Field(
        None,
        description="Target product for Y in buy-X-get-Y rewards. Required for *_off_y types.",
    )
    quantity: Optional[int] = Field(
        None,
        ge=1,
        description="Number of Y items the customer receives. Required for *_off_y and free_gift.",
    )

    # ------ Free Gift fields ------
    gift_product: Optional[GiftProductTarget] = Field(
        None,
        description="Gift product resolution metadata. Required for free_gift type.",
    )

    @model_validator(mode="after")
    def check_reward_fields(self):
        """Enforce field requirements based on reward type."""
        t = self.type

        # --- Tiered simple rewards ---
        if t in (RewardType.percentage_off, RewardType.fixed_amount_off, RewardType.free_gift_product):
            if self.value is None:
                raise ValueError(f"'value' is required for reward type '{t.value}'")

        # --- Buy X Get Y rewards ---
        elif t in (RewardType.percentage_off_y, RewardType.fixed_amount_off_y):
            if self.value is None:
                raise ValueError(f"'value' is required for reward type '{t.value}'")
            if self.y_target is None:
                raise ValueError(f"'y_target' is required for reward type '{t.value}'")
            if self.quantity is None:
                raise ValueError(f"'quantity' is required for reward type '{t.value}'")

        # --- Free gift (with resolution metadata) ---
        elif t == RewardType.free_gift:
            if self.gift_product is None:
                raise ValueError("'gift_product' is required for reward type 'free_gift'")
            if self.quantity is None:
                raise ValueError("'quantity' is required for reward type 'free_gift'")

        return self


# ---------------------------------------------------------------------------
# Tier Model (trigger + typed reward)
# ---------------------------------------------------------------------------

class TierModel(BaseModel):
    """A single tier object containing a trigger and optional reward."""

    trigger: TriggerModel
    reward: Optional[RewardModel] = Field(
        None, description="Reward configuration. Optional during trigger-only schema check."
    )


# ---------------------------------------------------------------------------
# Promotion State Model for current project
# ---------------------------------------------------------------------------

class PromotionState(BaseModel):
    message: str
    history: List[Dict[str, str]] = Field(default_factory=list)
    feature: Optional[str] = None
    tiers: List[Any] = Field(default_factory=list)
    tier_behavior: Optional[str] = None
    customer_eligibility: List[Any] = Field(default_factory=list)
    status: Optional[str] = None
    blockers: List[Any] = Field(default_factory=list)
    missing_fields: List[Any] = Field(default_factory=list)
    clarification_attempts: int = 0
    thread_id: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    # Dictionary compatibility layer to avoid breaking downstream LangGraph node code
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def keys(self):
        return self.model_fields.keys()