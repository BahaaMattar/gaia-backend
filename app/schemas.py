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

class Prediction(BaseModel):
    disease: str
    probability: float = Field(..., ge=0.0, le=1.0)

class AssessmentResponse(BaseModel):
    risk_level: RiskLevel
    probability: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    recommendation: str
    top_3_predictions: List[Prediction] = Field(..., min_items=1, max_items=3)


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


class GoogleAuthRequest(BaseModel):
    id_token: str = Field(..., min_length=20, max_length=4096)
    age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[Gender] = None
    phone: Optional[str] = Field(None, max_length=30)
    location: Optional[str] = Field(None, max_length=80)


class GoogleAccessAuthRequest(BaseModel):
    access_token: str = Field(..., min_length=20, max_length=4096)
    age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[Gender] = None
    phone: Optional[str] = Field(None, max_length=30)
    location: Optional[str] = Field(None, max_length=80)


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    age: Optional[int] = None
    gender: Optional[Gender] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    created_at: Optional[str] = None
    role: str
    is_active: bool


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


# Admin dashboard summary response
class AdminDashboardSummary(BaseModel):
    total_users: int
    inactive_users: int
    total_doctors: int
    active_doctors: int
    inactive_doctors: int
    total_admins: int
    total_assessments: int
    assessments_today: int
    new_users_this_week: int

# Doctor list item response
class DoctorListItem(BaseModel):
    id: int
    name: str
    email: str
    phone: Optional[str] = None
    location: Optional[str] = None
    role: str
    is_active: bool
    created_at: Optional[str] = None
    specialty: Optional[str] = None

# Update doctor status request
class UpdateDoctorStatusRequest(BaseModel):
    is_active: bool

# Full user list item (for admin user management)
class UserListItem(BaseModel):
    id: int
    name: str
    email: str
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    role: str
    is_active: bool
    created_at: Optional[str] = None
    specialty: Optional[str] = None

# Create any user with a chosen role
class CreateUserAdminRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)
    email: str = Field(..., min_length=5, max_length=254)
    password: str = Field(..., min_length=8, max_length=128)
    role: Literal["user", "doctor", "admin"] = "user"
    phone: Optional[str] = Field(None, max_length=30)
    location: Optional[str] = Field(None, max_length=80)
    specialty: Optional[str] = Field(None, max_length=100)

# Change any user's role
class UpdateUserRoleRequest(BaseModel):
    role: Literal["user", "doctor", "admin"]


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=254)


class ResetPasswordRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=254)
    code: str = Field(..., min_length=4, max_length=12)
    new_password: str = Field(..., min_length=8, max_length=128)

class StepSyncRequest(BaseModel):
    steps: int

class WaterSyncRequest(BaseModel):
    intake_ml: int