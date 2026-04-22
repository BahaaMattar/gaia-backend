from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import User, ContactThread, ContactMessage
from app.schemas import (
    ContactMessageIn,
    ContactMessageOut,
    ContactThreadOut,
    ContactThreadSummary,
)
from app.dependencies import get_user_from_header

router = APIRouter(prefix="/contact", tags=["Contact"])

_DAILY_MESSAGE_LIMIT = 5


def require_admin(current_user: User = Depends(get_user_from_header)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return current_user


def _thread_out(thread: ContactThread) -> ContactThreadOut:
    return ContactThreadOut(
        id=thread.id,
        status=thread.status,
        created_at=str(thread.created_at),
        updated_at=str(thread.updated_at),
        messages=[
            ContactMessageOut(
                id=m.id,
                sender_type=m.sender_type,
                content=m.content,
                created_at=str(m.created_at),
            )
            for m in thread.messages
        ],
    )


def _count_user_messages_last_24h(db: Session, thread: ContactThread) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    return (
        db.query(ContactMessage)
        .filter(
            ContactMessage.thread_id == thread.id,
            ContactMessage.sender_type == "user",
            ContactMessage.created_at >= cutoff,
        )
        .count()
    )


def _seconds_until_slot_opens(db: Session, thread: ContactThread) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    oldest_in_window = (
        db.query(ContactMessage)
        .filter(
            ContactMessage.thread_id == thread.id,
            ContactMessage.sender_type == "user",
            ContactMessage.created_at >= cutoff,
        )
        .order_by(ContactMessage.created_at.asc())
        .first()
    )
    if not oldest_in_window or oldest_in_window.created_at is None:
        return 0
    ts = oldest_in_window.created_at
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    opens_at = ts + timedelta(hours=24)
    remaining = (opens_at - datetime.now(timezone.utc)).total_seconds()
    return max(0, int(remaining))


# ── User endpoints ─────────────────────────────────────────────────────────────

@router.post("/threads", response_model=ContactThreadOut)
def create_thread(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_user_from_header),
):
    existing = (
        db.query(ContactThread)
        .filter(
            ContactThread.user_id == current_user.id,
            ContactThread.status == "open",
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=400,
            detail="You already have an open conversation. "
                   "Please continue there or wait until it is closed.",
        )

    thread = ContactThread(user_id=current_user.id, status="open")
    db.add(thread)
    db.commit()
    db.refresh(thread)
    return _thread_out(thread)


@router.get("/threads/me", response_model=ContactThreadOut)
def get_my_thread(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_user_from_header),
):
    thread = (
        db.query(ContactThread)
        .filter(ContactThread.user_id == current_user.id)
        .order_by(ContactThread.created_at.desc())
        .first()
    )
    if not thread:
        raise HTTPException(status_code=404, detail="No conversation found.")
    return _thread_out(thread)


@router.post("/threads/{thread_id}/messages", response_model=ContactMessageOut)
def send_message(
    thread_id: int,
    payload: ContactMessageIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_user_from_header),
):
    thread = db.query(ContactThread).filter(ContactThread.id == thread_id).first()
    if not thread or thread.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    if thread.status == "closed":
        raise HTTPException(status_code=400, detail="This conversation has been closed.")

    count = _count_user_messages_last_24h(db, thread)
    if count >= _DAILY_MESSAGE_LIMIT:
        retry_after = _seconds_until_slot_opens(db, thread)
        raise HTTPException(
            status_code=429,
            detail={
                "code": "message_limit_reached",
                "message": f"You have sent {_DAILY_MESSAGE_LIMIT} messages in the last 24 hours.",
                "retry_after_seconds": retry_after,
            },
        )

    msg = ContactMessage(
        thread_id=thread.id,
        sender_type="user",
        content=payload.content.strip(),
    )
    db.add(msg)
    thread.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(msg)
    return ContactMessageOut(
        id=msg.id,
        sender_type=msg.sender_type,
        content=msg.content,
        created_at=str(msg.created_at),
    )


# ── Admin endpoints ────────────────────────────────────────────────────────────

@router.get("/threads", response_model=list[ContactThreadSummary])
def list_threads(
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    threads = (
        db.query(ContactThread)
        .order_by(ContactThread.status.asc(), ContactThread.updated_at.desc())
        .all()
    )
    result = []
    for t in threads:
        msgs = t.messages
        last_msg = msgs[-1].content[:80] if msgs else None
        result.append(
            ContactThreadSummary(
                id=t.id,
                status=t.status,
                user_id=t.user_id,
                user_name=t.user.name,
                user_email=t.user.email,
                created_at=str(t.created_at),
                updated_at=str(t.updated_at),
                message_count=len(msgs),
                last_message=last_msg,
            )
        )
    return result


@router.get("/threads/{thread_id}/messages", response_model=ContactThreadOut)
def get_thread_messages(
    thread_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    thread = db.query(ContactThread).filter(ContactThread.id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return _thread_out(thread)


@router.post("/threads/{thread_id}/reply", response_model=ContactMessageOut)
def admin_reply(
    thread_id: int,
    payload: ContactMessageIn,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    thread = db.query(ContactThread).filter(ContactThread.id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    if thread.status == "closed":
        raise HTTPException(status_code=400, detail="Cannot reply to a closed conversation.")

    msg = ContactMessage(
        thread_id=thread.id,
        sender_type="admin",
        content=payload.content.strip(),
    )
    db.add(msg)
    thread.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(msg)
    return ContactMessageOut(
        id=msg.id,
        sender_type=msg.sender_type,
        content=msg.content,
        created_at=str(msg.created_at),
    )


@router.put("/threads/{thread_id}/close")
def close_thread(
    thread_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    thread = db.query(ContactThread).filter(ContactThread.id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    thread.status = "closed"
    thread.updated_at = datetime.now(timezone.utc)
    db.commit()
    return {"status": "closed", "thread_id": thread_id}
