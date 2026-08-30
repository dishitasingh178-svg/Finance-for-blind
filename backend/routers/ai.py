import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.models import User, ConversationSession
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
    Supports multi-turn conversational context and follow-up clarification resolution.

    Execution Flow:
    1. Validates user existence and ownership in SQLite.
    2. Resolves or creates a ConversationSession with strict user isolation and TTL checks.
    3. Injects conversation history and pending clarification context into the AI pipeline.
    4. Deterministic engine computes authoritative facts from SQLite database.
    5. Grounded explainer verbalizes the facts with strict anti-hallucination verification.
    6. Updates conversation state in SQLite and returns grounded response.
    """
    # 1. User existence validation
    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id {payload.user_id} not found.",
        )

    # 2. Resolve or create ConversationSession
    session: ConversationSession
    if payload.conversation_id:
        existing_session = db.query(ConversationSession).filter(
            ConversationSession.id == payload.conversation_id
        ).first()

        if not existing_session or existing_session.user_id != payload.user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation session '{payload.conversation_id}' not found.",
            )

        session = existing_session

        # Check TTL expiration (15 minutes)
        if session.is_expired():
            session.status = "expired"
            session.intent = None
            session.parameters = {}
            session.missing_parameters = []
            session.last_clarification_question = None
            session.last_user_query = None
    else:
        new_conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        session = ConversationSession(
            id=new_conv_id,
            user_id=payload.user_id,
            status="active",
            parameters={},
            missing_parameters=[],
        )
        db.add(session)
        db.flush()

    # 3. Build context for AI pipeline
    pipeline_context = session.to_context_dict()
    if payload.context:
        pipeline_context.update(payload.context)

    # 4. Execute AI pipeline with the caller's active database session
    try:
        pipeline_result = run_finSight_pipeline(
            user_id=payload.user_id,
            query=payload.query,
            db=db,
            context=pipeline_context,
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process financial AI query: {str(e)}",
        )

    # 5. Update and persist conversation state
    session_status = pipeline_result.get("conversation_status", "active")
    session.status = session_status
    session.last_user_query = payload.query
    session.updated_at = datetime.utcnow()

    if session_status == "awaiting_clarification":
        session.intent = pipeline_result.get("intent") or session.intent
        session.parameters = pipeline_result.get("parameters") or session.parameters or {}
        session.missing_parameters = pipeline_result.get("missing_parameters") or []
        session.last_clarification_question = pipeline_result.get("clarification_question")
    else:
        # Completed / active turn
        session.intent = pipeline_result.get("intent") or session.intent
        session.parameters = pipeline_result.get("parameters") or session.parameters or {}
        session.missing_parameters = []
        session.last_clarification_question = None

    try:
        db.commit()
    except Exception:
        db.rollback()

    return AskResponse(
        answer_text=pipeline_result.get("answer_text", "I don't have that information available."),
        structured_data=pipeline_result.get("structured_data", {}),
        conversation_id=session.id,
        conversation_status=session.status,
        execution_mode=pipeline_result.get("execution_mode", "REAL_LLM"),
    )


