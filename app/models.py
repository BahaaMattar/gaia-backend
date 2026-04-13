from sqlalchemy import Column, Integer, String, Float, DateTime, Date, ForeignKey, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .db import Base

class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    symptoms = relationship("SymptomEntry", back_populates="assessment", cascade="all, delete-orphan")
    result = relationship("PredictionResult", back_populates="assessment", uselist=False, cascade="all, delete-orphan")

class SymptomEntry(Base):
    __tablename__ = "symptom_entries"

    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, ForeignKey("assessments.id"), nullable=False)
    name = Column(String, nullable=False)
    severity = Column(Integer, nullable=True)
    duration_days = Column(Integer, nullable=True)

    assessment = relationship("Assessment", back_populates="symptoms")

class PredictionResult(Base):
    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, ForeignKey("assessments.id"), nullable=False, unique=True)

    risk_level = Column(String, nullable=False)
    probability = Column(Float, nullable=False)
    explanation = Column(String, nullable=False)
    recommendation = Column(String, nullable=False)

    assessment = relationship("Assessment", back_populates="result")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    location = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    reset_code_hash = Column(String, nullable=True)
    reset_expires_at = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    role = Column(String, nullable=False, default="user")
    is_active = Column(Boolean, nullable=False, default=True)
    specialty = Column(String, nullable=True)

class StepRecord(Base):
    __tablename__ = "step_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    steps = Column(Integer, nullable=False)
    record_date = Column(Date, server_default=func.current_date())


class WaterRecord(Base):
    __tablename__ = "water_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    intake_ml = Column(Integer, nullable=False)
    record_date = Column(Date, server_default=func.current_date())


class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    condition = Column(String, nullable=True)
    status = Column(String, nullable=False, default="pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", foreign_keys=[user_id], backref="appointments_as_user")
    doctor = relationship("User", foreign_keys=[doctor_id], backref="appointments_as_doctor")
    messages = relationship("Message", back_populates="appointment", cascade="all, delete-orphan", order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    appointment_id = Column(Integer, ForeignKey("appointments.id"), nullable=False, index=True)
    sender_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    appointment = relationship("Appointment", back_populates="messages")
    sender = relationship("User", foreign_keys=[sender_id])