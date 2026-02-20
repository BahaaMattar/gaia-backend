from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./app.db"  # creates app.db in repo root

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_user_columns(engine):
    with engine.begin() as conn:
        try:
            rows = conn.exec_driver_sql("PRAGMA table_info(users)").fetchall()
        except Exception:
            return

        existing = {row[1] for row in rows}
        if not existing:
            return

        columns = {
            "age": "INTEGER",
            "gender": "TEXT",
            "phone": "TEXT",
            "location": "TEXT",
            "reset_code_hash": "TEXT",
            "reset_expires_at": "INTEGER",
        }

        for name, ddl_type in columns.items():
            if name not in existing:
                conn.exec_driver_sql(f"ALTER TABLE users ADD COLUMN {name} {ddl_type}")
