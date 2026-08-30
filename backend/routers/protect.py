"""
FinSight Protect Router — AI Scam & Fraud Safety Checker
=========================================================

Endpoints for pattern-based SMS and message scam evaluation.
Separated strictly from the deterministic financial engine.
"""

from fastapi import APIRouter, HTTPException, status

from backend.schemas import ScamCheckRequest, ScamCheckResponse
from ai.scam_checker import assess_scam_message

router = APIRouter(tags=["Protect (Scam & Fraud Safety)"])


@router.post(
    "/protect/scam-check",
    response_model=ScamCheckResponse,
    summary="Assess message for scam & fraud patterns",
    description=(
        "Pattern-based LLM safety assessment for suspicious SMS or messages. "
        "Evaluates urgency, phishing links, credential requests, fake prizes, and impersonation. "
        "Grounded strictly in the supplied text."
    ),
)
@router.post(
    "/api/v1/protect/scam-check",
    response_model=ScamCheckResponse,
    include_in_schema=False,
)
def check_message_for_scams(payload: ScamCheckRequest) -> ScamCheckResponse:
    """
    Evaluates pasted message text for scam characteristics.
    Returns structured risk level, specific indicators, grounded explanation,
    recommended safety actions, and clear non-deterministic limitations.
    """
    try:
        result = assess_scam_message(message=payload.message)
        return ScamCheckResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assess message safety: {str(e)}",
        )
