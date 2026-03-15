from fastapi import FastAPI, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
from dotenv import load_dotenv

import joblib
import numpy as np
import pandas as pd

import os

from .schemas import (
    AssessmentRequest,
    AssessmentResponse,
    SignUpRequest,
    LoginRequest,
    GoogleAuthRequest,
    GoogleAccessAuthRequest,
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
from app.dependencies import get_user_from_header
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests
import requests
import time
import secrets

ENV_PATH = Path(__file__).resolve().parent.parent / os.getenv("GAIA_ENV_FILE", ".env.local")
load_dotenv(ENV_PATH)

app = FastAPI(title="GAIA Backend")

# Include routers
from app.routers.admin_router import router as admin_router
app.include_router(admin_router)

# Load ML models once at startup
MODEL_PATH = "app/ml-results/"
model_pipeline = joblib.load(os.path.join(MODEL_PATH, "hybrid_model.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_PATH, "disease_label_encoder.pkl"))
selected_symptoms = joblib.load(os.path.join(MODEL_PATH, "selected_symptoms.pkl"))

# Load original feature names (all symptom columns from training data)
df_training = pd.read_csv('app/Data/Diseases_and_Symptoms_dataset.csv')
all_symptom_columns = [col for col in df_training.columns if col != 'diseases']

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


def _allowed_google_client_ids() -> list[str]:
    return [
        value.strip()
        for value in os.getenv("GAIA_GOOGLE_CLIENT_IDS", "").split(",")
        if value.strip()
    ]


def _assert_google_audience(audience: str | None):
    allowed_client_ids = _allowed_google_client_ids()
    if allowed_client_ids and audience not in allowed_client_ids:
        raise HTTPException(status_code=401, detail="Google client id is not allowed")


def _verify_google_identity(token: str) -> dict:
    request = google_requests.Request()
    try:
        identity = google_id_token.verify_oauth2_token(
            token,
            request,
            audience=None,
        )
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid Google token") from exc

    issuer = identity.get("iss")
    if issuer not in ("accounts.google.com", "https://accounts.google.com"):
        raise HTTPException(status_code=401, detail="Invalid Google token issuer")

    _assert_google_audience(identity.get("aud"))

    if not identity.get("email_verified"):
        raise HTTPException(status_code=401, detail="Google email is not verified")

    if not identity.get("email"):
        raise HTTPException(status_code=400, detail="Google token missing email")

    return identity


def _verify_google_access_identity(access_token: str) -> dict:
    try:
        token_info = requests.get(
            "https://www.googleapis.com/oauth2/v3/tokeninfo",
            params={"access_token": access_token},
            timeout=10,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=401, detail="Unable to verify Google access token") from exc

    if token_info.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid Google access token")

    token_info_data = token_info.json()
    audience = token_info_data.get("aud") or token_info_data.get("azp")
    _assert_google_audience(audience)

    try:
        user_info = requests.get(
            "https://openidconnect.googleapis.com/v1/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=401, detail="Unable to fetch Google profile") from exc

    if user_info.status_code != 200:
        raise HTTPException(status_code=401, detail="Unable to fetch Google profile")

    identity = user_info.json()
    if not identity.get("email"):
        raise HTTPException(status_code=400, detail="Google token missing email")
    if not identity.get("email_verified"):
        raise HTTPException(status_code=401, detail="Google email is not verified")

    identity["aud"] = audience
    return identity


def _build_auth_response(user: User) -> dict:
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
            "role": user.role,
            "is_active": user.is_active,
        },
    }


def _find_or_create_google_user(
    *,
    identity: dict,
    db: Session,
    age: int | None,
    gender: str | None,
    phone: str | None,
    location: str | None,
) -> User:
    email = str(identity["email"]).strip().lower()
    user = db.query(User).filter(User.email == email).first()
    if user:
        return user

    display_name = str(identity.get("name") or email.split("@", 1)[0]).strip()
    user = User(
        name=display_name[:80] if display_name else "Google User",
        email=email,
        password_hash=hash_password(secrets.token_urlsafe(32)),
        age=age,
        gender=gender,
        phone=phone.strip() if phone else None,
        location=location.strip() if location else None,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

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

    # 3) GET ML PREDICTION
    # Build feature vector matching your training data (all 230 original symptoms)
    symptom_names = [s.name for s in payload.symptoms]
    
    # Create binary feature vector with ALL original symptoms (1 if present, 0 otherwise)
    feature_vector = np.zeros(len(all_symptom_columns))
    matched_symptoms = []
    unmatched_symptoms = []
    
    for i, symptom_col in enumerate(all_symptom_columns):
        if symptom_col.lower() in [s.lower() for s in symptom_names]:
            feature_vector[i] = 1
            matched_symptoms.append(symptom_col)
    
    # Track unmatched symptoms for debugging
    for symptom in symptom_names:
        if not any(symptom.lower() == col.lower() for col in all_symptom_columns):
            unmatched_symptoms.append(symptom)
    
    # Debug log
    print(f"\n=== ML PREDICTION DEBUG ===")
    print(f"Symptoms received from frontend: {symptom_names}")
    print(f"Matched symptoms ({len(matched_symptoms)}): {matched_symptoms}")
    if unmatched_symptoms:
        print(f"UNMATCHED symptoms ({len(unmatched_symptoms)}): {unmatched_symptoms}")
    print(f"Feature vector sum: {int(feature_vector.sum())} symptoms active")
    print(f"=========================\n")
    
    # Get probabilities
    y_probs = model_pipeline.predict_proba([feature_vector])[0]
    top_3_indices = np.argsort(y_probs)[-3:][::-1]
    
    # Get top disease with highest probability
    top_disease_idx = top_3_indices[0]
    top_disease = label_encoder.classes_[top_disease_idx]
    probability = float(y_probs[top_disease_idx])
    
    # Categorize risk level
    if probability > 0.7:
        risk_level = "High"
    elif probability > 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    
    explanation = f"Based on your symptoms, the model predicts: {top_disease} (confidence: {probability:.1%})"
    recommendation = "Please consult a healthcare professional for accurate diagnosis."

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
        role="user",
        is_active=True,
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
            "role": user.role,
            "is_active": user.is_active,
        },
    }


@app.post("/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is inactive. Please contact admin.")
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
            "role": user.role,
            "is_active": user.is_active,
        },
    }


@app.post("/auth/google", response_model=AuthResponse)
def google_auth(payload: GoogleAuthRequest, db: Session = Depends(get_db)):
    identity = _verify_google_identity(payload.id_token)
    user = _find_or_create_google_user(
        identity=identity,
        db=db,
        age=payload.age,
        gender=payload.gender,
        phone=payload.phone,
        location=payload.location,
    )
    return _build_auth_response(user)


@app.post("/auth/google/access", response_model=AuthResponse)
def google_auth_with_access_token(
    payload: GoogleAccessAuthRequest,
    db: Session = Depends(get_db),
):
    identity = _verify_google_access_identity(payload.access_token)
    user = _find_or_create_google_user(
        identity=identity,
        db=db,
        age=payload.age,
        gender=payload.gender,
        phone=payload.phone,
        location=payload.location,
    )
    return _build_auth_response(user)


@app.get("/auth/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_user_from_header)):
    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
        "age": current_user.age,
        "gender": current_user.gender,
        "phone": current_user.phone,
        "location": current_user.location,
        "created_at": str(current_user.created_at),
        "role": current_user.role,
        "is_active": current_user.is_active,
    }


@app.put("/auth/me", response_model=UserResponse)
def update_me(
    payload: UpdateUserRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_user_from_header),
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
        "role": current_user.role,
        "is_active": current_user.is_active,
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
