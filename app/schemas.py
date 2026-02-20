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


class SignUpRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)
    email: str = Field(..., min_length=5, max_length=254)
    password: str = Field(..., min_length=8, max_length=128)
    age: int = Field(..., ge=0, le=120)
    gender: Gender
    phone: Optional[str] = Field(None, max_length=30)
    location: Optional[str] = Field(None, max_length=80)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=254)
    password: str = Field(..., min_length=8, max_length=128)


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    age: Optional[int] = None
    gender: Optional[Gender] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    created_at: Optional[str] = None


class AuthResponse(BaseModel):
    token: str
    user: UserResponse


class UpdateUserRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=80)
    email: Optional[str] = Field(None, min_length=5, max_length=254)
    password: Optional[str] = Field(None, min_length=8, max_length=128)
    age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[Gender] = None
    phone: Optional[str] = Field(None, max_length=30)
    location: Optional[str] = Field(None, max_length=80)


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=254)


class ResetPasswordRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=254)
    code: str = Field(..., min_length=4, max_length=12)
    new_password: str = Field(..., min_length=8, max_length=128)
