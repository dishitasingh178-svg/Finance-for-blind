"""
AI Conversational Assistant Router for FinSight.

Provides natural-language conversational financial assistant endpoints (POST /ask and POST /api/v1/ask).
Connects FastAPI -> AI Pipeline -> Deterministic Financial Engine -> Database.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.models import User
from backend.schemas import AskRequest, AskResponse
from ai.pipeline import run_finSight_pipeline

router = APIRouter(tags=["Conversational AI"])


@router.post("/ask", response_model=AskResponse, summary="Ask FinSight AI Assistant")
@router.post("/api/v1/ask", response_model=AskResponse, include_in_schema=False)
@router.post("/api/v1/voice/query", response_model=AskResponse, include_in_schema=False)
def ask_financial_copilot(
    payload: AskRequest,
    db: Session = Depends(get_db),
) -> AskResponse:
    """
    Handles natural language personal finance questions for visually impaired users.

    Execution Flow:
    1. Validates user existence and ownership in SQLite.
    2. Passes the user query, integer user_id, and active DB session to the AI pipeline.
    3. Intent Router maps query to exactly one deterministic financial engine function.
    4. Deterministic engine computes authoritative facts from SQLite database.
    5. Grounded explainer verbalizes the facts with strict anti-hallucination number verification.
    6. Returns natural language answer text and authoritative structured engine data.
    """
    # 1. User existence validation
    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id {payload.user_id} not found.",
        )

    # 2. Execute AI pipeline with the caller's active database session
    try:
        pipeline_result = run_finSight_pipeline(
            user_id=payload.user_id,
            query=payload.query,
            db=db,
            context=payload.context,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process financial AI query: {str(e)}",
        )

    return AskResponse(
        answer_text=pipeline_result.get("answer_text", "I don't have that information available."),
        structured_data=pipeline_result.get("structured_data", {}),
    )
