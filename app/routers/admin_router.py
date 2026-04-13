from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.models import User, Assessment
from app.schemas import (
    AdminDashboardSummary,
    DoctorListItem,
    UpdateDoctorStatusRequest,
    UserListItem,
    CreateUserAdminRequest,
    UpdateUserRoleRequest,
)
from app.db import get_db
from app.auth import hash_password

from app.dependencies import get_user_from_header


def require_admin(current_user: User = Depends(get_user_from_header)):
    if not current_user.is_active:
        raise HTTPException(status_code=403, detail="Account is inactive. Please contact admin.")
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return current_user


def _user_to_item(u: User) -> UserListItem:
    return UserListItem(
        id=u.id,
        name=u.name,
        email=u.email,
        age=u.age,
        gender=u.gender,
        phone=u.phone,
        location=u.location,
        role=u.role,
        is_active=u.is_active,
        created_at=str(u.created_at),
        specialty=u.specialty,
    )


router = APIRouter(prefix="/admin", tags=["Admin"])


# ── Dashboard ─────────────────────────────────────────────────────────────────

@router.get("/dashboard-summary", response_model=AdminDashboardSummary)
def admin_dashboard_summary(db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    total_users    = db.query(User).filter(User.role == "user").count()
    inactive_users = db.query(User).filter(User.role == "user",   User.is_active == False).count()
    total_doctors  = db.query(User).filter(User.role == "doctor").count()
    active_doctors = db.query(User).filter(User.role == "doctor", User.is_active == True).count()
    inactive_doctors = total_doctors - active_doctors
    total_admins   = db.query(User).filter(User.role == "admin").count()
    total_assessments = db.query(Assessment).count()

    today_utc    = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago_utc = datetime.now(timezone.utc) - timedelta(days=7)

    assessments_today  = db.query(Assessment).filter(Assessment.created_at >= today_utc).count()
    new_users_this_week = db.query(User).filter(
        User.created_at >= week_ago_utc,
        User.role == "user",
    ).count()

    return AdminDashboardSummary(
        total_users=total_users,
        inactive_users=inactive_users,
        total_doctors=total_doctors,
        active_doctors=active_doctors,
        inactive_doctors=inactive_doctors,
        total_admins=total_admins,
        total_assessments=total_assessments,
        assessments_today=assessments_today,
        new_users_this_week=new_users_this_week,
    )


# ── User management ───────────────────────────────────────────────────────────

@router.get("/users", response_model=list[UserListItem])
def admin_list_users(db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [_user_to_item(u) for u in users]


@router.post("/users", response_model=UserListItem)
def admin_create_user(payload: CreateUserAdminRequest, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    email = payload.email.strip().lower()
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(
        name=payload.name.strip(),
        email=email,
        password_hash=hash_password(payload.password),
        phone=payload.phone.strip() if payload.phone else None,
        location=payload.location.strip() if payload.location else None,
        role=payload.role,
        is_active=True,
        specialty=payload.specialty,
        latitude=payload.latitude,
        longitude=payload.longitude,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _user_to_item(user)


@router.patch("/users/{id}/role", response_model=UserListItem)
def admin_update_user_role(id: int, payload: UpdateUserRoleRequest, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    user = db.query(User).filter(User.id == id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot change your own role")
    user.role = payload.role
    db.commit()
    db.refresh(user)
    return _user_to_item(user)


@router.patch("/users/{id}/status", response_model=UserListItem)
def admin_update_user_status(id: int, payload: UpdateDoctorStatusRequest, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    user = db.query(User).filter(User.id == id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")
    user.is_active = payload.is_active
    db.commit()
    db.refresh(user)
    return _user_to_item(user)


@router.delete("/users/{id}")
def admin_delete_user(id: int, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    user = db.query(User).filter(User.id == id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    db.delete(user)
    db.commit()
    return {"status": "ok"}


# ── Doctor management (kept for backwards compatibility) ──────────────────────

@router.get("/doctors", response_model=list[DoctorListItem])
def admin_list_doctors(db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    doctors = db.query(User).filter(User.role == "doctor").all()
    return [
        DoctorListItem(
            id=d.id, name=d.name, email=d.email, phone=d.phone,
            location=d.location, role=d.role, is_active=d.is_active,
            created_at=str(d.created_at), specialty=d.specialty,
        ) for d in doctors
    ]


@router.patch("/doctors/{id}/status", response_model=DoctorListItem)
def admin_update_doctor_status(id: int, payload: UpdateDoctorStatusRequest, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    doctor = db.query(User).filter(User.id == id, User.role == "doctor").first()
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
    doctor.is_active = payload.is_active
    db.commit()
    db.refresh(doctor)
    return DoctorListItem(
        id=doctor.id, name=doctor.name, email=doctor.email, phone=doctor.phone,
        location=doctor.location, role=doctor.role, is_active=doctor.is_active,
        created_at=str(doctor.created_at),
    )
