"""
AI Intent Router for FinSight.

Parses natural language financial questions, voice transcripts, and commands
into structured, typed intent schemas.

ARCHITECTURAL PRINCIPLES:
- ZERO financial calculations (does not compute, round, or aggregate numbers).
- Supports Gemini Function/Tool Calling (REAL_LLM) with graceful degradation to
  deterministic regex parsing (MOCK_FALLBACK).
- Multi-turn clarification support (resolves follow-ups like '8k').
- Strict output schema: {"intent": str, "arguments": dict, "execution_mode": str}
"""

import re
import logging
from typing import Dict, Any, Optional

from ai.llm_client import LLMClient
from ai.conversation import conversation_manager, parse_shorthand_amount

logger = logging.getLogger(__name__)

CATEGORIES_MAP = {
    "food": "Food",
    "groceries": "Food",
    "grocery": "Food",
    "dining": "Food",
    "restaurant": "Food",
    "restaurants": "Food",
    "transport": "Transport",
    "travel": "Transport",
    "fuel": "Transport",
    "uber": "Transport",
    "cab": "Transport",
    "shopping": "Shopping",
    "clothes": "Shopping",
    "electronics": "Shopping",
    "bills": "Bills",
    "utilities": "Bills",
    "electricity": "Bills",
    "water": "Bills",
    "entertainment": "Entertainment",
    "movies": "Entertainment",
    "netflix": "Entertainment",
    "healthcare": "Healthcare",
    "medical": "Healthcare",
    "doctor": "Healthcare",
    "medicine": "Healthcare",
    "education": "Education",
    "college": "Education",
    "tuition": "Education",
    "books": "Education",
    "other": "Other",
}


def _extract_amount(text: str) -> Optional[str]:
    """Extracts numeric currency amount from text (e.g. ₹5,000, 8k, 8000 rupees, 500.50)."""
    # First check shorthand '8k'
    shorthand = parse_shorthand_amount(text)
    if shorthand:
        return shorthand

    # Look for currency prefix or suffix, or plain numbers
    patterns = [
        r"(?:₹|rs\.?|inr|rupees?)\s*([\d,]+(?:\.\d{1,2})?)",
        r"([\d,]+(?:\.\d{1,2})?)\s*(?:₹|rs\.?|inr|rupees?)",
        r"(?:for|cost(?:s|ing)?|amount of|spend|pay|send)\s*(?:₹|rs\.?|inr|rupees?)?\s*([\d,]+(?:\.\d{1,2})?)",
        r"\b([\d,]+(?:\.\d{1,2})?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw_val = match.group(1).replace(",", "").strip()
            try:
                val = float(raw_val)
                if val > 0:
                    return raw_val
            except ValueError:
                continue
    return None


def route_intent_fallback(query: str, confirmation_token: Optional[str] = None) -> Dict[str, Any]:
    """
    Deterministic rule-based intent router used in MOCK_FALLBACK mode.
    """
    if not query or not query.strip():
        return {"intent": "unknown", "arguments": {}}

    text = query.strip()
    lower_text = text.lower()

    # 1. Payment Execution / Confirmation
    confirm_keywords = ["confirm", "confirm payment", "yes", "proceed", "send it", "approve", "execute payment", "yes confirm", "yes send"]
    if lower_text in confirm_keywords or re.search(r"^(?:yes|confirm|proceed|send it)(?:\s+payment)?$", lower_text):
        token = confirmation_token
        id_match = re.search(r"(?:payment\s*(?:id|#)?|token)\s*(\d+)", lower_text)
        if id_match:
            token = id_match.group(1)
        return {
            "intent": "payment_execute",
            "arguments": {
                "pending_payment_id": token,
                "confirmation_token": token,
            },
        }

    # 2. Payment Preview ("Send ₹5,000 to Dr Rao", "Pay 1000 to Rahul", "Transfer 500 to mom")
    pay_patterns = [
        r"(?:send|pay|transfer)\s+(?:₹|rs\.?|inr|rupees?)?\s*([\d,]+(?:\.\d{1,2})?)\s*(?:₹|rs\.?|inr|rupees?)?\s+to\s+([a-zA-Z0-9\s\.\-_]+)",
        r"(?:send|pay|transfer)\s+([a-zA-Z0-9\s\.\-_]+)\s+(?:₹|rs\.?|inr|rupees?)?\s*([\d,]+(?:\.\d{1,2})?)\s*(?:₹|rs\.?|inr|rupees?)?",
    ]
    for pattern in pay_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            g1, g2 = match.group(1).strip(), match.group(2).strip()
            amt_candidate = g1.replace(",", "")
            try:
                float(amt_candidate)
                amount = amt_candidate
                recipient = g2
            except ValueError:
                amt_candidate = g2.replace(",", "")
                amount = amt_candidate
                recipient = g1

            recipient = re.sub(r"\b(rupees?|inr|rs|please|now)\b", "", recipient, flags=re.IGNORECASE).strip()

            return {
                "intent": "payment_preview",
                "arguments": {
                    "amount": amount,
                    "recipient_name": recipient,
                },
            }

    # 3. Affordability Check ("Can I afford headphones for 8000?", "Can I buy a bike for 12000?", "Can I afford it?")
    if any(k in lower_text for k in ["afford", "can i buy", "can i purchase", "can i spend", "should i buy"]):
        amount = _extract_amount(text)
        item_match = re.search(r"(?:afford|buy|purchase)\s+(?:a|an|the)?\s*([a-zA-Z0-9\s]+?)(?:\s+(?:for|costing|worth|at)|$)", text, re.IGNORECASE)
        item_name = item_match.group(1).strip() if item_match else None
        if item_name and re.match(r"^[\d,]+$", item_name):
            item_name = None

        return {
            "intent": "check_affordability",
            "arguments": {
                "amount": amount,
                "item_name": item_name,
            },
        }

    # 4. Spending Summary ("How much did I spend on food this month?", "Spending summary", "What did I spend last month?")
    if any(k in lower_text for k in ["spend", "spending", "spent", "expenses", "expense summary"]):
        period = "last_month" if "last month" in lower_text or "previous month" in lower_text else "this_month"
        matched_cat = None
        for word, cat_name in CATEGORIES_MAP.items():
            if re.search(r"\b" + word + r"\b", lower_text):
                matched_cat = cat_name
                break

        return {
            "intent": "get_spending_summary",
            "arguments": {
                "period": period,
                "category": matched_cat,
            },
        }

    # 5. Goal Projections ("When will I finish my Emergency Fund?", "Project goal", "When will I reach my goal?")
    if any(k in lower_text for k in ["goal", "emergency fund", "save for", "savings target", "reach my target", "finish my"]):
        goal_name = None
        goal_match = re.search(r"(?:finish|reach|complete|for|project)\s+(?:my\s+)?([a-zA-Z0-9\s]+?)(?:\s+goal|\s+target|\?|$)", text, re.IGNORECASE)
        if goal_match:
            candidate = goal_match.group(1).strip()
            if candidate.lower() not in ["the", "my", "a", "this"]:
                goal_name = candidate

        hypo_match = re.search(r"(?:if i save|contribute|put in)\s+(?:₹|rs\.?|inr)?\s*([\d,]+)", lower_text)
        hypo_contrib = hypo_match.group(1).replace(",", "") if hypo_match else None

        return {
            "intent": "project_goal_completion",
            "arguments": {
                "goal_name": goal_name,
                "hypothetical_contribution": hypo_contrib,
            },
        }

    # 6. Insights ("Any insights?", "What are my insights?", "Show alerts", "Financial updates")
    if any(k in lower_text for k in ["insight", "insights", "alerts", "updates", "anomalies", "recommendation", "trends"]):
        return {
            "intent": "get_insights",
            "arguments": {},
        }

    # 7. Balance Check ("How much money do I have?", "What is my balance?", "Account balance", "Check balance")
    if any(k in lower_text for k in ["balance", "money do i have", "how much i have", "funds", "account total", "what do i have"]):
        return {
            "intent": "get_balance",
            "arguments": {},
        }

    return {
        "intent": "unknown",
        "arguments": {"raw_query": text},
    }


def route_intent(
    query: str,
    confirmation_token: Optional[str] = None,
    conversation_id: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """
    Classifies natural language text into a structured intent and arguments.
    Attempts REAL_LLM first (if configured), falling back gracefully to deterministic MOCK_FALLBACK.

    Supported intents:
    - get_balance
    - get_spending_summary
    - check_affordability
    - project_goal_completion
    - get_insights
    - payment_preview
    - payment_execute
    """
    if not query or not query.strip():
        return {
            "intent": "unknown",
            "arguments": {},
            "execution_mode": "MOCK_FALLBACK",
            "provider_error": None,
        }

    # Step 1: Check if this resolves an active clarification in multi-turn conversation
    resolved_clarification = conversation_manager.resolve_clarification(conversation_id, query)
    if resolved_clarification:
        resolved_clarification["execution_mode"] = "MOCK_FALLBACK"
        resolved_clarification["provider_error"] = None
        return resolved_clarification

    # Step 2: Try Gemini Tool Calling (REAL_LLM) if available
    client = llm_client or LLMClient()
    if client.is_available():
        history = conversation_manager.get_llm_messages(conversation_id)
        intent_data, err = client.call_tool_router(query=query, history=history)
        if intent_data:
            # If confirmation_token was provided in the request, inject it into payment_execute args
            if confirmation_token and intent_data.get("intent") == "payment_execute":
                if not intent_data.get("arguments", {}).get("pending_payment_id"):
                    intent_data.setdefault("arguments", {})["pending_payment_id"] = confirmation_token
                    intent_data["arguments"]["confirmation_token"] = confirmation_token

            intent_data["execution_mode"] = "REAL_LLM"
            intent_data["provider_error"] = None
            return intent_data
        else:
            logger.warning(f"LLM tool routing failed, degrading to MOCK_FALLBACK: {err}")
            fallback_res = route_intent_fallback(query=query, confirmation_token=confirmation_token)
            fallback_res["execution_mode"] = "MOCK_FALLBACK"
            fallback_res["provider_error"] = err
            return fallback_res

    # Step 3: Local deterministic rule routing (MOCK_FALLBACK)
    fallback_res = route_intent_fallback(query=query, confirmation_token=confirmation_token)
    fallback_res["execution_mode"] = "MOCK_FALLBACK"
    fallback_res["provider_error"] = None
    return fallback_res
