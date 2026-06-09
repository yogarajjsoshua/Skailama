from typing import Dict, Any, Optional, List, Annotated
from pydantic import BaseModel, Field, field_validator
from enum import Enum

# --- Recruitment / Query Models from User reference ---

class QueryResponse(BaseModel):
    content: str
    chat_id: str
    message_id: str
    role: str
    timestamp: str
    requires_input: bool = False
    workflow_state: Optional[str] = None
    ui: Optional[Dict[str, Any]] = None


class RecruitmentState(BaseModel):
    user_id: str
    chat_id: str
    message_id: str
    query: str
    intent: Optional[str] = None
    workflow: Optional[str] = None
    current_node: str = "intent_discovery"
    context: Dict[str, Any] = Field(default_factory=dict)
    response: str = ""
    requires_input: bool = False
    selected_customer_id: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class CandidateType(int, Enum):
    ALL = 0
    ACTIVE = 1
    INACTIVE = 2


class TalentPoolPayload(BaseModel):
    skills: List[str] = Field(default_factory=list, description="List of required skills")
    candidate_id: Optional[str] = Field(None, alias="candidateID", description="Specific candidate ID")
    search_query: Optional[str] = Field(None, alias="searchQuery", description="Search query string")
    count: int = Field(default=10, ge=1, le=100, description="Number of candidates to return")
    page: int = Field(default=0, ge=0, description="Page number for pagination")
    customer_id: Optional[str] = Field(None, alias="customerID", description="Customer ID filter")
    job_id: Optional[str] = Field(None, alias="jobID", description="Job ID filter")
    created_by: List[str] = Field(default_factory=list, alias="createdBy", description="List of creator IDs")
    type: CandidateType = Field(default=CandidateType.ALL, description="Candidate type filter")

    @field_validator('skills', mode='before')
    def parse_skills(cls, v):
        if isinstance(v, str):
            return [skill.strip() for skill in v.split(',') if skill.strip()]
        return v or []

    @field_validator('created_by', mode='before')
    def parse_created_by(cls, v):
        if isinstance(v, str):
            return [v.strip() for v in v.split(',') if v.strip()]
        return v or []

    def to_api_dict(self) -> dict:
        """Convert to API format with correct field names"""
        return {
            "skills": self.skills,
            "candidateID": self.candidate_id,
            "searchQuery": self.search_query,
            "count": self.count,
            "page": self.page,
            "customerID": self.customer_id,
            "jobID": self.job_id,
            "createdBy": self.created_by,
            "type": self.type.value,
        }


# --- Promotion State Model for current project ---

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