from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.models import User, Assessment
from app.schemas import AdminDashboardSummary, CreateDoctorRequest, DoctorListItem, UpdateDoctorStatusRequest
from app.db import get_db
from app.auth import hash_password

# Admin role guard dependency
from app.dependencies import get_user_from_header

def require_admin(current_user: User = Depends(get_user_from_header)):
    if not current_user.is_active:
        raise HTTPException(status_code=403, detail="Account is inactive. Please contact admin.")
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return current_user

router = APIRouter(prefix="/admin", tags=["Admin"])

@router.get("/dashboard-summary", response_model=AdminDashboardSummary)
def admin_dashboard_summary(db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    total_users = db.query(User).count()
    total_doctors = db.query(User).filter(User.role == "doctor").count()
    active_doctors = db.query(User).filter(User.role == "doctor", User.is_active == True).count()
    total_assessments = db.query(Assessment).count()
    return AdminDashboardSummary(
        total_users=total_users,
        total_doctors=total_doctors,
        active_doctors=active_doctors,
        total_assessments=total_assessments,
    )

@router.get("/doctors", response_model=list[DoctorListItem])
def admin_list_doctors(db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    doctors = db.query(User).filter(User.role == "doctor").all()
    return [
        DoctorListItem(
            id=d.id,
            name=d.name,
            email=d.email,
            phone=d.phone,
            location=d.location,
            role=d.role,
            is_active=d.is_active,
            created_at=str(d.created_at),
        ) for d in doctors
    ]

@router.post("/doctors", response_model=DoctorListItem)
def admin_create_doctor(payload: CreateDoctorRequest, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    email = payload.email.strip().lower()
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    doctor = User(
        name=payload.name.strip(),
        email=email,
        password_hash=hash_password(payload.password),
        phone=payload.phone.strip() if payload.phone else None,
        location=payload.location.strip() if payload.location else None,
        role="doctor",
        is_active=True,
    )
    db.add(doctor)
    db.commit()
    db.refresh(doctor)
    return DoctorListItem(
        id=doctor.id,
        name=doctor.name,
        email=doctor.email,
        phone=doctor.phone,
        location=doctor.location,
        role=doctor.role,
        is_active=doctor.is_active,
        created_at=str(doctor.created_at),
    )

@router.patch("/doctors/{id}/status", response_model=DoctorListItem)
def admin_update_doctor_status(id: int, payload: UpdateDoctorStatusRequest, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    doctor = db.query(User).filter(User.id == id, User.role == "doctor").first()
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
    doctor.is_active = payload.is_active
    db.commit()
    db.refresh(doctor)
    return DoctorListItem(
        id=doctor.id,
        name=doctor.name,
        email=doctor.email,
        phone=doctor.phone,
        location=doctor.location,
        role=doctor.role,
        is_active=doctor.is_active,
        created_at=str(doctor.created_at),
    )
