"""
Goals Router for FinSight.
Provides endpoints for goal tracking and projections.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.db import get_db

router = APIRouter(prefix="/api/v1/goals", tags=["Goals"])


@router.get("/")
def list_goals(db: Session = Depends(get_db)):
    """
    Returns list of financial goals.
    (Endpoint scaffold for the foundation step).
    """
    return {
        "status": "active",
        "goals": [],
        "message": "FinSight Goals API scaffold. Full calculation integration planned for Day 2.",
    }
