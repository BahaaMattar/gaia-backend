from math import radians, cos, sin, asin, sqrt
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.dependencies import get_user_from_header
from app.models import User, Appointment, Message
from app.schemas import (
    NearbyDoctorResponse,
    AppointmentCreate,
    AppointmentResponse,
    MessageCreate,
    MessageResponse,
)

router = APIRouter(tags=["Doctors & Appointments"])

# ── Disease → Specialty mapping ───────────────────────────────────────────────

_KEYWORD_SPECIALTY: list[tuple[list[str], str]] = [
    (["cold", "flu", "influenza", "fever", "cough", "runny", "congestion", "sneez", "strep", "sore throat", "body ache", "chills", "malaise", "fatigue", "tired", "weakness", "upper respiratory", "rhinit"], "General Practitioner / Family Medicine"),
    (["heart", "cardiac", "hypertension", "blood pressure", "coronary", "arrhythmia"], "Cardiologist"),
    (["lung", "pulmonary", "asthma", "bronchitis", "pneumonia", "copd", "tuberculosis", "respiratory", "wheez", "sleep apnea"], "Pulmonologist / Sleep Specialist"),
    (["neurol", "migraine", "stroke", "epilepsy", "seizure", "parkinson", "alzheimer", "brain", "vertigo", "neuropathy"], "Neurologist"),
    (["skin", "derma", "eczema", "psoriasis", "acne", "rash", "urticaria", "melanoma"], "Dermatologist"),
    (["stomach", "gastro", "hepat", "liver", "bowel", "colon", "irritable", "crohn", "ulcer", "acid reflux", "gerd", "nausea", "diarrhea"], "Gastroenterologist / Hepatologist / Proctologist"),
    (["diabetes", "thyroid", "endocrin", "hormone", "adrenal", "pituitary", "metabolic"], "Endocrinologist"),
    (["kidney", "renal", "nephr"], "Nephrologist"),
    (["joint", "arthritis", "rheumat", "orthoped", "bone", "fracture", "ligament", "tendon", "spine", "back pain", "osteoporosis"], "Orthopedic Surgeon / Rheumatologist"),
    (["ear", "nose", "throat", "sinus", "tonsil", "laryngit", "pharyngit", "hearing", "otit", "ent", "nasal polyp"], "Otolaryngologist (ENT)"),
    (["allerg", "immunolog", "anaphylaxis", "hay fever", "hives"], "Allergist / Immunologist"),
    (["eye", "vision", "ophthal", "glaucoma", "cataract", "retina"], "Ophthalmologist"),
    (["urin", "bladder", "prostate", "urology", "kidney stone"], "Urologist"),
    (["blood", "anemia", "leukemia", "lymphoma", "hematol"], "Hematologist"),
    (["mental", "psychiatr", "depress", "anxiety", "bipolar", "schizo", "ocd", "ptsd", "panic"], "Psychiatrist"),
    (["child", "pediatr", "infant", "newborn"], "Pediatrician"),
    (["gynecol", "obstet", "menstrual", "uterus", "ovary", "pregnancy", "cervix"], "Gynecologist / Obstetrician (OB/GYN)"),
    (["tooth", "dental", "gum", "perio", "oral cavity"], "Dentist / Periodontist"),
]

_FALLBACK_SPECIALTY = "General Surgery / Critical Care / Pain Management"


def _condition_to_specialty(condition: Optional[str]) -> Optional[str]:
    if not condition:
        return None
    lower = condition.lower()
    for keywords, specialty in _KEYWORD_SPECIALTY:
        if any(kw in lower for kw in keywords):
            return specialty
    return None


# ── Haversine distance ────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/doctors/nearby", response_model=list[NearbyDoctorResponse])
def get_nearby_doctors(
    lat: float = Query(..., ge=-90, le=90),
    lng: float = Query(..., ge=-180, le=180),
    condition: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None),
    radius_km: float = Query(50.0, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Return active doctors within radius_km, preferring matching specialty."""
    target_specialty = _condition_to_specialty(condition)
    is_high_risk = (risk_level or "").lower() == "high"

    doctors = (
        db.query(User)
        .filter(
            User.role == "doctor",
            User.is_active == True,
            User.latitude.isnot(None),
            User.longitude.isnot(None),
        )
        .all()
    )

    def _is_match(specialty: str | None) -> bool:
        if not target_specialty or not specialty:
            return False
        return any(p.strip().lower() in specialty.lower() for p in target_specialty.split("/"))

    results: list[NearbyDoctorResponse] = []
    for d in doctors:
        dist = _haversine_km(lat, lng, d.latitude, d.longitude)
        if dist <= radius_km:
            results.append(
                NearbyDoctorResponse(
                    id=d.id,
                    name=d.name,
                    email=d.email,
                    phone=d.phone,
                    specialty=d.specialty,
                    latitude=d.latitude,
                    longitude=d.longitude,
                    distance_km=round(dist, 2),
                    is_active=d.is_active,
                    is_specialty_match=_is_match(d.specialty),
                )
            )

    def sort_key(r: NearbyDoctorResponse):
        specialty_match = 0 if r.is_specialty_match else 1
        is_urgent = any(kw in (r.specialty or "").lower() for kw in ["urgent", "emergency", "internal medicine"])
        urgency_boost = 0 if (is_high_risk and is_urgent) else 1
        return (min(specialty_match, urgency_boost), r.distance_km)

    results.sort(key=sort_key)
    return results


@router.post("/appointments", response_model=AppointmentResponse)
def create_appointment(
    payload: AppointmentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_user_from_header),
):
    if current_user.role not in ("user", "admin"):
        raise HTTPException(status_code=403, detail="Only users can book appointments.")

    doctor = db.query(User).filter(User.id == payload.doctor_id, User.role == "doctor").first()
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found.")
    if not doctor.is_active:
        raise HTTPException(status_code=400, detail="Doctor is not available.")

    existing = (
        db.query(Appointment)
        .filter(
            Appointment.user_id == current_user.id,
            Appointment.doctor_id == payload.doctor_id,
            Appointment.status == "pending",
        )
        .first()
    )
    if existing:
        raise HTTPException(status_code=409, detail="You already have a pending appointment with this doctor.")

    appt = Appointment(
        user_id=current_user.id,
        doctor_id=payload.doctor_id,
        condition=payload.condition,
        status="pending",
    )
    db.add(appt)
    db.commit()
    db.refresh(appt)

    return AppointmentResponse(
        id=appt.id,
        user_id=appt.user_id,
        doctor_id=appt.doctor_id,
        condition=appt.condition,
        status=appt.status,
        created_at=str(appt.created_at),
        user_name=current_user.name,
        doctor_name=doctor.name,
        doctor_specialty=doctor.specialty,
    )


@router.get("/appointments/my", response_model=list[AppointmentResponse])
def get_my_appointments(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_user_from_header),
):
    """Returns appointments for the logged-in user (as patient)."""
    appts = (
        db.query(Appointment)
        .filter(Appointment.user_id == current_user.id)
        .order_by(Appointment.created_at.desc())
        .all()
    )
    result = []
    for a in appts:
        doctor = db.query(User).filter(User.id == a.doctor_id).first()
        result.append(
            AppointmentResponse(
                id=a.id,
                user_id=a.user_id,
                doctor_id=a.doctor_id,
                condition=a.condition,
                status=a.status,
                created_at=str(a.created_at),
                user_name=current_user.name,
                doctor_name=doctor.name if doctor else None,
                doctor_specialty=doctor.specialty if doctor else None,
            )
        )
    return result


@router.get("/doctor/appointments", response_model=list[AppointmentResponse])
def get_doctor_appointments(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_user_from_header),
):
    """Returns appointments booked with the logged-in doctor."""
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can access this endpoint.")

    appts = (
        db.query(Appointment)
        .filter(Appointment.doctor_id == current_user.id)
        .order_by(Appointment.created_at.desc())
        .all()
    )
    result = []
    for a in appts:
        patient = db.query(User).filter(User.id == a.user_id).first()
        result.append(
            AppointmentResponse(
                id=a.id,
                user_id=a.user_id,
                doctor_id=a.doctor_id,
                condition=a.condition,
                status=a.status,
                created_at=str(a.created_at),
                user_name=patient.name if patient else None,
                doctor_name=current_user.name,
                doctor_specialty=current_user.specialty,
            )
        )
    return result


@router.patch("/appointments/{appointment_id}/status", response_model=AppointmentResponse)
def update_appointment_status(
    appointment_id: int,
    status: str = Query(..., pattern="^(pending|confirmed|completed|cancelled)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_user_from_header),
):
    """Doctor can confirm/complete/cancel; user can cancel their own."""
    appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found.")

    is_doctor = current_user.role == "doctor" and appt.doctor_id == current_user.id
    is_patient = appt.user_id == current_user.id

    if not is_doctor and not is_patient:
        raise HTTPException(status_code=403, detail="Not authorized.")

    if is_patient and status not in ("cancelled",):
        raise HTTPException(status_code=403, detail="Patients can only cancel appointments.")

    appt.status = status
    db.commit()
    db.refresh(appt)

    doctor = db.query(User).filter(User.id == appt.doctor_id).first()
    patient = db.query(User).filter(User.id == appt.user_id).first()

    return AppointmentResponse(
        id=appt.id,
        user_id=appt.user_id,
        doctor_id=appt.doctor_id,
        condition=appt.condition,
        status=appt.status,
        created_at=str(appt.created_at),
        user_name=patient.name if patient else None,
        doctor_name=doctor.name if doctor else None,
        doctor_specialty=doctor.specialty if doctor else None,
    )


@router.get("/appointments/{appointment_id}/messages", response_model=list[MessageResponse])
def get_messages(
    appointment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_user_from_header),
):
    appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found.")

    if current_user.id not in (appt.user_id, appt.doctor_id):
        raise HTTPException(status_code=403, detail="Not authorized.")

    msgs = (
        db.query(Message)
        .filter(Message.appointment_id == appointment_id)
        .order_by(Message.created_at)
        .all()
    )
    result = []
    for m in msgs:
        sender = db.query(User).filter(User.id == m.sender_id).first()
        result.append(
            MessageResponse(
                id=m.id,
                appointment_id=m.appointment_id,
                sender_id=m.sender_id,
                sender_name=sender.name if sender else "Unknown",
                content=m.content,
                created_at=str(m.created_at),
            )
        )
    return result


@router.post("/appointments/{appointment_id}/messages", response_model=MessageResponse)
def send_message(
    appointment_id: int,
    payload: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_user_from_header),
):
    appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found.")

    if current_user.id not in (appt.user_id, appt.doctor_id):
        raise HTTPException(status_code=403, detail="Not authorized.")

    if appt.status == "cancelled":
        raise HTTPException(status_code=400, detail="Cannot message on a cancelled appointment.")

    msg = Message(
        appointment_id=appointment_id,
        sender_id=current_user.id,
        content=payload.content.strip(),
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)

    return MessageResponse(
        id=msg.id,
        appointment_id=msg.appointment_id,
        sender_id=msg.sender_id,
        sender_name=current_user.name,
        content=msg.content,
        created_at=str(msg.created_at),
    )
