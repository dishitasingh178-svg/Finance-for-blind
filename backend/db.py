"""
Database configuration and session management for FinSight.
Uses SQLite with SQLAlchemy ORM.
"""

import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./finsight.db")

# SQLite-specific connect args (check_same_thread=False for FastAPI multi-threaded requests)
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=os.getenv("SQL_ECHO", "false").lower() == "true",
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session per request.
    Creates a session, yields it, and ensures it is closed after request completion.
    Does not auto-commit.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initializes the database by creating all tables defined in models.
    """
    # Import all models to register them on Base.metadata
    from backend import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
