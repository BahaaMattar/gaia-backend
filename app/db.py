import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import declarative_base, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_override = os.getenv("GAIA_ENV_FILE")
candidate_env_paths = (
    [PROJECT_ROOT / env_override]
    if env_override
    else [PROJECT_ROOT / ".env.local", PROJECT_ROOT / ".env" / ".env.local"]
)

for env_path in candidate_env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    searched_paths = ", ".join(str(path) for path in candidate_env_paths)
    raise RuntimeError(
        "DATABASE_URL environment variable is not set. "
        "Set it in .env.local or GAIA_ENV_FILE. "
        f"(searched env files: {searched_paths})"
    )

parsed_db_url = make_url(DATABASE_URL)
backend = parsed_db_url.get_backend_name()
if backend != "postgresql":
    raise RuntimeError(
        "DATABASE_URL must use PostgreSQL for Supabase "
        f"(got backend: {backend})."
    )

require_supabase_host = os.getenv("GAIA_REQUIRE_SUPABASE_HOST", "true").lower() in (
    "1",
    "true",
    "yes",
)
db_host = (parsed_db_url.host or "").lower()
if require_supabase_host and "supabase" not in db_host:
    raise RuntimeError(
        "DATABASE_URL must point to a Supabase host when "
        "GAIA_REQUIRE_SUPABASE_HOST is enabled."
    )

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
