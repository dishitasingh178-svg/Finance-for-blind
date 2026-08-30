"""
Conversation session model for FinSight multi-turn conversational context.

Tracks conversation states, pending clarifications, extracted entities,
and session lifecycles across conversational turns.
"""

from datetime import datetime, timezone
from typing import Set, Optional, Dict, Any, List
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship, validates
from backend.db import Base

VALID_CONVERSATION_STATUSES: Set[str] = {
    "active",
    "awaiting_clarification",
    "completed",
    "expired",
}

DEFAULT_CONVERSATION_TTL_SECONDS = 900  # 15 minutes


class ConversationSession(Base):
    """
    Persists conversational state for a user across multiple dialogue turns.
    """
    __tablename__ = "conversation_sessions"

    id = Column(String(64), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    intent = Column(String(100), nullable=True)  # check_affordability, project_goal_completion, etc.
    parameters = Column(JSON, default=dict, nullable=False)
    missing_parameters = Column(JSON, default=list, nullable=False)
    last_clarification_question = Column(Text, nullable=True)
    last_user_query = Column(Text, nullable=True)
    status = Column(String(50), default="active", nullable=False)  # active, awaiting_clarification, completed, expired
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User")

    @validates("status")
    def validate_status(self, key: str, value: str) -> str:
        if value not in VALID_CONVERSATION_STATUSES:
            raise ValueError(f"Invalid conversation status '{value}'. Must be one of: {VALID_CONVERSATION_STATUSES}")
        return value

    def is_expired(self, ttl_seconds: int = DEFAULT_CONVERSATION_TTL_SECONDS) -> bool:
        """Returns True if the conversation has been inactive for longer than ttl_seconds."""
        if not self.updated_at:
            return False
        now = datetime.utcnow()
        elapsed = (now - self.updated_at).total_seconds()
        return elapsed > ttl_seconds

    def to_context_dict(self) -> Dict[str, Any]:
        """Serializes current session state into a context dictionary for AI pipeline."""
        return {
            "conversation_id": self.id,
            "user_id": self.user_id,
            "intent": self.intent,
            "parameters": self.parameters or {},
            "missing_parameters": self.missing_parameters or [],
            "last_clarification_question": self.last_clarification_question,
            "last_user_query": self.last_user_query,
            "status": self.status,
        }

    def __repr__(self) -> str:
        return (
            f"<ConversationSession(id='{self.id}', user_id={self.user_id}, "
            f"intent='{self.intent}', status='{self.status}', updated_at={self.updated_at})>"
        )
