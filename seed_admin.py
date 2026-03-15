#!/usr/bin/env python3
"""
One-time admin seed script for GAIA FastAPI backend.

Usage: python seed_admin.py

This script will either:
1. Create a new admin user (admin@example.com) if none exists
2. Promote an existing user to admin if they already exist

For local development/FYP use only.
"""

from app.db import SessionLocal
from app.models import User
from app.auth import hash_password


def seed_admin():
    """Create or promote admin user."""
    db = SessionLocal()
    
    try:
        # Check if admin user already exists
        admin_email = "admin@example.com"
        existing_user = db.query(User).filter(User.email == admin_email).first()
        
        if existing_user:
            # User exists - promote/confirm as admin
            existing_user.role = "admin"
            existing_user.is_active = True
            db.commit()
            print(f"✅ Existing user promoted/confirmed as admin: {admin_email}")
        else:
            # User doesn't exist - create new admin
            admin_user = User(
                name="Admin",
                email=admin_email,
                password_hash=hash_password("12345678"),
                role="admin",
                is_active=True,
            )
            db.add(admin_user)
            db.commit()
            print(f"✅ Admin user created: {admin_email}")
            print("📋 Default password: 12345678")
            print("⚠️  Please change the password after first login!")
            
    except Exception as e:
        db.rollback()
        print(f"❌ Error: {e}")
        return False
        
    finally:
        db.close()
        
    return True


if __name__ == "__main__":
    print("🚀 GAIA Admin Seed Script")
    print("=" * 40)
    
    success = seed_admin()
    
    if success:
        print("\n✨ Admin setup completed successfully!")
    else:
        print("\n💥 Admin setup failed!")