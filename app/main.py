from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware

from .schemas import AssessmentRequest, AssessmentResponse
from .db import Base, engine, get_db
from .models import Assessment, SymptomEntry, PredictionResult

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
