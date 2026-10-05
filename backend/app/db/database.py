"""
Database setup.

Defaults to a local SQLite file at backend/data/tao_fin.db -- zero
external services to stand up. Swap to Postgres later just by
changing DATABASE_URL, e.g.:

    DATABASE_URL=postgresql://user:pass@localhost:5432/taofin

SQLAlchemy's engine/session API is the same either way, so nothing
else in the app needs to change.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

_DATA_DIR = Path("data")
_DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_DATA_DIR}/tao_fin.db")

# check_same_thread=False is only needed for SQLite (FastAPI may use
# the connection from a different thread than it was created on).
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Create all tables if they don't already exist. Safe to call every startup."""
    # Import models here so they're registered on Base before create_all runs.
    from app.db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a DB session, closes it after the request."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()