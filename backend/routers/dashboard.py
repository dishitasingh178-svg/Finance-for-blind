"""
Dashboard Router for FinSight.
Provides structured dashboard facts for the accessibility layer.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.db import get_db

router = APIRouter(prefix="/api/v1/dashboard", tags=["Dashboard"])


@router.get("/")
def get_dashboard_summary(db: Session = Depends(get_db)):
    """
    Returns structured dashboard overview facts.
    (Endpoint scaffold for the foundation step).
    """
    return {
        "status": "active",
        "message": "FinSight Dashboard API scaffold. Full calculation integration planned for Day 2.",
    }
