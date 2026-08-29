"""
AI Pipeline for FinSight.

Coordinates the end-to-end flow:
Query -> intent_router (Gemini or Fallback) -> backend dispatcher -> deterministic engine -> explainer.
"""

from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from ai.llm_client import LLMClient
from ai.intent_router import route_intent
from ai.explainer import explain
from ai.conversation import conversation_manager
from backend.engine.dispatcher import dispatch_intent


class AIPipeline:
    """
    End-to-end orchestrator connecting natural language queries to the backend dispatcher
    and formatting the response via the explainer with multi-turn session tracking.
    """

    @staticmethod
    def process_query(
        user_id: int,
        query: str,
        db: Session,
        confirmation_token: Optional[str] = None,
        conversation_id: Optional[str] = None,
        voice: bool = False,
        llm_client: Optional[LLMClient] = None,
    ) -> Dict[str, Any]:
        """
        Processes a user query end-to-end.

        1. Resolves multi-turn conversation context.
        2. Classifies intent via Gemini Tool Calling or deterministic fallback.
        3. Executes deterministic business logic via dispatcher with user isolation.
        4. Explains authoritative facts via grounded LLM or template explainer.
        5. Records conversation turn for follow-up clarifications.
        """
        active_conversation_id = conversation_id or f"user-{user_id}-session"

        # Step 1: Extract intent and slots (REAL_LLM or MOCK_FALLBACK)
        intent_data = route_intent(
            query=query,
            confirmation_token=confirmation_token,
            conversation_id=active_conversation_id,
            llm_client=llm_client,
        )

        exec_mode = intent_data.get("execution_mode", "MOCK_FALLBACK")

        # Step 2: Dispatch to deterministic backend (propagates security/validation ValueErrors)
        facts = dispatch_intent(
            user_id=user_id,
            intent_data=intent_data,
            db=db,
            confirmation_token=confirmation_token,
        )

        # Step 3: Explain authoritative facts
        result = explain(
            intent_data=intent_data,
            facts=facts,
            query=query,
            llm_client=llm_client,
            preferred_mode=exec_mode,
        )

        # Attach conversation session and consistency fields
        result["conversation_id"] = active_conversation_id
        result["structured_data"] = result.get("structured_facts", {})

        # Step 4: Record turn in conversation manager
        conversation_manager.record_turn(
            conversation_id=active_conversation_id,
            query=query,
            intent=intent_data.get("intent", "unknown"),
            arguments=intent_data.get("arguments", {}),
            facts=facts,
            answer_text=result.get("answer_text", ""),
        )

        return result
