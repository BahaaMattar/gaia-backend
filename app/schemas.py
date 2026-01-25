from pydantic import BaseModel, Field
from typing import List, Literal, Optional

Gender = Literal["male", "female", "other"]
RiskLevel = Literal["Low", "Medium", "High"]

class Symptom(BaseModel):
    name: str = Field(..., min_length=1)
    severity: Optional[int] = Field(None, ge=1, le=5)
    duration_days: Optional[int] = Field(None, ge=0, le=3650)

class AssessmentRequest(BaseModel):
    age: int = Field(..., ge=0, le=120)
    gender: Gender
    symptoms: List[Symptom] = Field(..., min_length=1)

class AssessmentResponse(BaseModel):
    risk_level: RiskLevel
    probability: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    recommendation: str
