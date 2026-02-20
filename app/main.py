from fastapi import FastAPI, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from dotenv import load_dotenv
import os

from .schemas import (
    AssessmentRequest,
    AssessmentResponse,
    SignUpRequest,
    LoginRequest,
    AuthResponse,
    UserResponse,
    UpdateUserRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
)
from .db import Base, engine, get_db, ensure_user_columns
from .models import Assessment, SymptomEntry, PredictionResult, User
from .auth import (
    hash_password,
    verify_password,
    create_token,
    decode_token,
    generate_reset_code,
    hash_reset_code,
    verify_reset_code,
    RESET_CODE_TTL_SECONDS,
)
from .email_service import send_email
import time

ENV_PATH = Path(__file__).resolve().parent.parent / os.getenv("GAIA_ENV_FILE", ".env.local")
load_dotenv(ENV_PATH)

app = FastAPI(title="GAIA Backend")

# Allow CORS for local development (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # simplest for local dev
    allow_credentials=False,      # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables automatically (simple local-only)
Base.metadata.create_all(bind=engine)
ensure_user_columns(engine)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/assessments", response_model=AssessmentResponse)
def create_assessment(payload: AssessmentRequest, db: Session = Depends(get_db)):
    # 1) Create assessment row
    assessment = Assessment(age=payload.age, gender=payload.gender)
    db.add(assessment)
    db.flush()  # get assessment.id

    # 2) Save symptom entries
    for s in payload.symptoms:
        db.add(SymptomEntry(
            assessment_id=assessment.id,
            name=s.name,
            severity=s.severity,
            duration_days=s.duration_days
        ))

    # 3) Dummy result for now (later: ML)
    risk_level = "Low"
    probability = 0.2
    explanation = "Based on the provided symptoms, the risk appears to be low."
    recommendation = "Monitor your symptoms and consult a doctor if they worsen."

    db.add(PredictionResult(
        assessment_id=assessment.id,
        risk_level=risk_level,
        probability=probability,
        explanation=explanation,
        recommendation=recommendation
    ))

    db.commit()

    return {
        "assessment_id": assessment.id,
        "risk_level": risk_level,
        "probability": probability,
        "explanation": explanation,
        "recommendation": recommendation
    }


def _get_user_from_header(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid auth header")

    token = authorization.split(" ", 1)[1].strip()
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token subject")

    return user


@app.post("/auth/signup", response_model=AuthResponse)
def signup(payload: SignUpRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        name=payload.name.strip(),
        email=email,
        password_hash=hash_password(payload.password),
        age=payload.age,
        gender=payload.gender,
        phone=payload.phone.strip() if payload.phone else None,
        location=payload.location.strip() if payload.location else None,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_token(user.id)
    return {
        "token": token,
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "age": user.age,
            "gender": user.gender,
            "phone": user.phone,
            "location": user.location,
            "created_at": str(user.created_at),
        },
    }


@app.post("/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_token(user.id)
    return {
        "token": token,
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "age": user.age,
            "gender": user.gender,
            "phone": user.phone,
            "location": user.location,
            "created_at": str(user.created_at),
        },
    }


@app.get("/auth/me", response_model=UserResponse)
def get_me(current_user: User = Depends(_get_user_from_header)):
    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
        "age": current_user.age,
        "gender": current_user.gender,
        "phone": current_user.phone,
        "location": current_user.location,
        "created_at": str(current_user.created_at),
    }


@app.put("/auth/me", response_model=UserResponse)
def update_me(
    payload: UpdateUserRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(_get_user_from_header),
):
    if payload.email:
        email = payload.email.strip().lower()
        existing = (
            db.query(User)
            .filter(User.email == email, User.id != current_user.id)
            .first()
        )
        if existing:
            raise HTTPException(status_code=409, detail="Email already registered")
        current_user.email = email

    if payload.name:
        current_user.name = payload.name.strip()

    if payload.password:
        current_user.password_hash = hash_password(payload.password)

    if payload.age is not None:
        current_user.age = payload.age

    if payload.gender:
        current_user.gender = payload.gender

    if payload.phone is not None:
        current_user.phone = payload.phone.strip() if payload.phone else None

    if payload.location is not None:
        current_user.location = payload.location.strip() if payload.location else None

    db.commit()
    db.refresh(current_user)

    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
        "age": current_user.age,
        "gender": current_user.gender,
        "phone": current_user.phone,
        "location": current_user.location,
        "created_at": str(current_user.created_at),
    }


@app.post("/auth/forgot")
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    debug = os.getenv("GAIA_DEBUG_ERRORS", "false").lower() in ("1", "true", "yes")
    email = payload.email.strip().lower()
    user = db.query(User).filter(User.email == email).first()

    if user:
        code = generate_reset_code()
        user.reset_code_hash = hash_reset_code(code)
        user.reset_expires_at = int(time.time()) + RESET_CODE_TTL_SECONDS
        db.commit()

        try:
            send_email(
                to_email=user.email,
                subject="GAIA password reset code",
                body=(
                    f"Your password reset code is {code}.\n\n"
                    "It expires in 15 minutes. If you did not request this, ignore this email."
                ),
            )
        except Exception as exc:
            user.reset_code_hash = None
            user.reset_expires_at = None
            db.commit()
            raise HTTPException(
                status_code=500,
                detail=str(exc) if debug else "Email send failed",
            )

    return {"status": "ok"}


@app.post("/auth/reset")
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    user = db.query(User).filter(User.email == email).first()
    if not user or not user.reset_code_hash or not user.reset_expires_at:
        raise HTTPException(status_code=400, detail="Invalid reset request")

    if user.reset_expires_at < int(time.time()):
        raise HTTPException(status_code=400, detail="Reset code expired")

    if not verify_reset_code(payload.code.strip(), user.reset_code_hash):
        raise HTTPException(status_code=400, detail="Invalid reset code")

    user.password_hash = hash_password(payload.new_password)
    user.reset_code_hash = None
    user.reset_expires_at = None
    db.commit()

    return {"status": "ok"}

@app.get("/assessments/{assessment_id}")
def get_assessment(assessment_id: int, db: Session = Depends(get_db)):
    assessment = db.query(Assessment).filter(Assessment.id == assessment_id).first()
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")

    result = assessment.result
    return {
        "assessment": {
            "id": assessment.id,
            "age": assessment.age,
            "gender": assessment.gender,
            "created_at": str(assessment.created_at)
        },
        "symptoms": [
            {"name": s.name, "severity": s.severity, "duration_days": s.duration_days}
            for s in assessment.symptoms
        ],
        "result": None if not result else {
            "risk_level": result.risk_level,
            "probability": result.probability,
            "explanation": result.explanation,
            "recommendation": result.recommendation
        }
    }
