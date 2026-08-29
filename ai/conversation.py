"""
Multi-turn Conversation and Clarification Context Manager for FinSight.

Maintains in-memory conversation history and resolves follow-up clarification answers
(e.g., Turn 1: "Can I afford it?" -> clarification_needed -> Turn 2: "8k" -> resolved amount).
"""

import re
from datetime import datetime
from typing import Dict, Any, Optional, List


def parse_shorthand_amount(text: str) -> Optional[str]:
    """
    Parses currency amounts including shorthand like '8k', '8.5k', '₹8,000', '8000'.
    """
    if not text:
        return None

    cleaned = text.strip().lower()

    # Match '8k', '8.5k', '8 k'
    k_match = re.search(r"(?:₹|rs\.?|inr)?\s*([\d]+(?:\.\d+)?)\s*k\b", cleaned, re.IGNORECASE)
    if k_match:
        try:
            val = float(k_match.group(1)) * 1000
            return f"{val:.2f}" if val % 1 != 0 else str(int(val))
        except ValueError:
            pass

    # Match standard numbers
    num_match = re.search(r"(?:₹|rs\.?|inr)?\s*([\d,]+(?:\.\d{1,2})?)\s*(?:₹|rs\.?|inr|rupees?)?", cleaned)
    if num_match:
        val_str = num_match.group(1).replace(",", "").strip()
        try:
            val = float(val_str)
            if val > 0:
                return val_str
        except ValueError:
            pass

    return None


class ConversationManager:
    """
    In-memory session state store for multi-turn conversations and clarifications.
    """

    def __init__(self):
        # Maps conversation_id -> list of turn dicts
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}

    def get_history(self, conversation_id: Optional[str]) -> List[Dict[str, Any]]:
        """Returns turn history for a conversation."""
        if not conversation_id or conversation_id not in self._sessions:
            return []
        return self._sessions[conversation_id]

    def get_llm_messages(self, conversation_id: Optional[str], max_turns: int = 4) -> List[Dict[str, str]]:
        """Returns message list for LLM context."""
        history = self.get_history(conversation_id)
        messages = []
        for turn in history[-max_turns:]:
            messages.append({"role": "user", "content": turn.get("query", "")})
            if turn.get("answer_text"):
                messages.append({"role": "assistant", "content": turn.get("answer_text", "")})
        return messages

    def get_last_turn(self, conversation_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Returns the most recent turn for the conversation."""
        history = self.get_history(conversation_id)
        return history[-1] if history else None

    def resolve_clarification(
        self,
        conversation_id: Optional[str],
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Attempts to resolve a pending clarification request using the user's follow-up input.

        Example:
        - Prior: intent=check_affordability, missing amount
        - User: "8k"
        - Resolved: intent=check_affordability, arguments={"amount": "8000", ...}
        """
        last_turn = self.get_last_turn(conversation_id)
        if not last_turn:
            return None

        facts = last_turn.get("facts", {})
        if facts.get("status") != "clarification_needed":
            return None

        prior_intent = last_turn.get("intent")
        prior_args = last_turn.get("arguments", {})

        # 1. Affordability Missing Amount Clarification
        if prior_intent == "check_affordability":
            amount = parse_shorthand_amount(query)
            if amount:
                item_name = prior_args.get("item_name")
                return {
                    "intent": "check_affordability",
                    "arguments": {
                        "amount": amount,
                        "item_name": item_name,
                    },
                }

        # 2. Goal Name Clarification
        elif prior_intent == "project_goal_completion":
            candidate_name = query.strip()
            # If query is not a new intent keyword
            if not any(k in candidate_name.lower() for k in ["balance", "spend", "pay", "send", "insight"]):
                hypo = prior_args.get("hypothetical_contribution")
                return {
                    "intent": "project_goal_completion",
                    "arguments": {
                        "goal_name": candidate_name,
                        "hypothetical_contribution": hypo,
                    },
                }

        # 3. Payment Missing Amount or Recipient Clarification
        elif prior_intent == "payment_preview":
            missing_amt = not prior_args.get("amount")
            missing_rec = not prior_args.get("recipient_name")

            if missing_amt and not missing_rec:
                amt = parse_shorthand_amount(query)
                if amt:
                    return {
                        "intent": "payment_preview",
                        "arguments": {
                            "amount": amt,
                            "recipient_name": prior_args.get("recipient_name"),
                        },
                    }
            elif missing_rec and not missing_amt:
                rec = query.strip()
                return {
                    "intent": "payment_preview",
                    "arguments": {
                        "amount": prior_args.get("amount"),
                        "recipient_name": rec,
                    },
                }

        return None

    def record_turn(
        self,
        conversation_id: str,
        query: str,
        intent: str,
        arguments: Dict[str, Any],
        facts: Dict[str, Any],
        answer_text: str,
    ) -> None:
        """Records a completed turn in the conversation session."""
        if not conversation_id:
            return

        if conversation_id not in self._sessions:
            self._sessions[conversation_id] = []

        self._sessions[conversation_id].append({
            "query": query,
            "intent": intent,
            "arguments": arguments,
            "facts": facts,
            "answer_text": answer_text,
            "timestamp": datetime.utcnow(),
        })

    def clear(self, conversation_id: Optional[str] = None) -> None:
        """Clears sessions (for testing)."""
        if conversation_id:
            self._sessions.pop(conversation_id, None)
        else:
            self._sessions.clear()


# Global in-memory singleton
conversation_manager = ConversationManager()
