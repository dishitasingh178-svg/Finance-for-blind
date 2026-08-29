"""
AI Explainer for FinSight.

Translates authoritative deterministic backend facts into accessible, concise,
screen-reader-friendly natural language narrations.

STRICT ARCHITECTURAL BOUNDARY:
- The Explainer MUST NEVER calculate or modify financial values.
- All numbers, balances, percentages, and metrics come strictly from the structured facts.
- Supports Gemini Grounded Narration (REAL_LLM) with graceful fallback to deterministic
  template narration (MOCK_FALLBACK).
- Assigns appropriate ARIA live region priorities ("polite" vs "assertive").
"""

import logging
from decimal import Decimal
from datetime import datetime, date
from typing import Dict, Any, Optional, List

from ai.llm_client import LLMClient

logger = logging.getLogger(__name__)


def _format_currency(val: Any) -> str:
    """Formats monetary values with Indian Rupee symbol and commas."""
    if val is None:
        return "₹0.00"
    try:
        dec = Decimal(str(val))
        return f"₹{dec:,.2f}"
    except Exception:
        return f"₹{val}"


def _format_date(val: Any) -> str:
    """Formats datetime or date into spoken-friendly format."""
    if val is None:
        return ""
    if isinstance(val, (datetime, date)):
        return val.strftime("%B %d, %Y")
    try:
        dt = datetime.fromisoformat(str(val))
        return dt.strftime("%B %d, %Y")
    except Exception:
        return str(val)


def explain_fallback(
    intent_data: Dict[str, Any],
    facts: Dict[str, Any],
    query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deterministic template-based explainer guaranteeing exact arithmetic fidelity.
    """
    if not isinstance(facts, dict):
        facts = {}

    intent_name = intent_data.get("intent", facts.get("intent", "unknown"))
    status = facts.get("status")

    # 1. Clarification Needed
    if status == "clarification_needed":
        question = facts.get("question", "Could you please provide more details?")
        return {
            "intent": intent_name,
            "answer_text": question,
            "aria_priority": "polite",
            "requires_confirmation": False,
            "confirmation_token": None,
            "pending_payment_id": None,
            "structured_facts": facts,
            "structured_data": facts,
            "conversation_status": "clarification_needed",
        }

    # 2. Unsupported Intent
    if status == "unsupported_intent" or intent_name == "unknown":
        return {
            "intent": "unknown",
            "answer_text": (
                "I'm sorry, I didn't quite catch that. You can ask me about your balance, "
                "spending by category, affordability for a purchase, savings goals, "
                "financial insights, or initiate a payment."
            ),
            "aria_priority": "polite",
            "requires_confirmation": False,
            "confirmation_token": None,
            "pending_payment_id": None,
            "structured_facts": facts,
            "structured_data": facts,
            "conversation_status": "completed",
        }

    # =========================================================================
    # get_balance
    # =========================================================================
    if intent_name == "get_balance":
        balance_str = _format_currency(facts.get("balance", "0.00"))
        as_of_str = _format_date(facts.get("as_of"))
        date_part = f" as of {as_of_str}" if as_of_str else ""
        answer = f"Your current authoritative balance is {balance_str}{date_part}."
        return {
            "intent": "get_balance",
            "answer_text": answer,
            "aria_priority": "polite",
            "requires_confirmation": False,
            "confirmation_token": None,
            "pending_payment_id": None,
            "structured_facts": facts,
            "structured_data": facts,
            "conversation_status": "completed",
        }

    # =========================================================================
    # get_spending_summary
    # =========================================================================
    elif intent_name == "get_spending_summary":
        period_str = "this month" if facts.get("period") == "this_month" else "last month"
        by_cat = facts.get("by_category", {})
        vs_pct = facts.get("vs_last_period_pct", {})
        req_cat = facts.get("requested_category")

        if req_cat and req_cat in by_cat:
            cat_amt = _format_currency(by_cat[req_cat])
            pct_val = vs_pct.get(req_cat, Decimal("0.00"))
            if pct_val > Decimal("0.00"):
                trend = f", which is up {pct_val:.2f}% compared to last month"
            elif pct_val < Decimal("0.00"):
                trend = f", which is down {abs(pct_val):.2f}% compared to last month"
            else:
                trend = ""
            answer = f"You spent {cat_amt} on {req_cat} {period_str}{trend}."
        else:
            total_str = _format_currency(facts.get("total", "0.00"))
            top_parts = []
            for cat, amt in sorted(by_cat.items(), key=lambda x: Decimal(str(x[1])), reverse=True)[:3]:
                if Decimal(str(amt)) > Decimal("0.00"):
                    top_parts.append(f"{cat}: {_format_currency(amt)}")
            breakdown = f" Main categories include {', '.join(top_parts)}." if top_parts else ""
            answer = f"Your total spending {period_str} is {total_str}.{breakdown}"

        return {
            "intent": "get_spending_summary",
            "answer_text": answer,
            "aria_priority": "polite",
            "requires_confirmation": False,
            "confirmation_token": None,
            "pending_payment_id": None,
            "structured_facts": facts,
            "structured_data": facts,
            "conversation_status": "completed",
        }

    # =========================================================================
    # check_affordability
    # =========================================================================
    elif intent_name == "check_affordability":
        can_afford = facts.get("can_afford", False)
        amt_str = _format_currency(facts.get("amount", "0.00"))
        item_str = f" for {facts.get('item_name')}" if facts.get("item_name") else ""
        bal_after_str = _format_currency(facts.get("balance_after", "0.00"))
        bills_str = _format_currency(facts.get("upcoming_bills", "0.00"))
        impact_months = facts.get("savings_goal_impact_months", Decimal("0"))

        if can_afford:
            goal_note = ""
            if impact_months and Decimal(str(impact_months)) > 0:
                goal_note = f" Note that this may delay your savings goals by {impact_months} month(s)."
            answer = (
                f"Yes, you can afford this purchase{item_str} of {amt_str}. "
                f"Your balance after the purchase will be {bal_after_str}, "
                f"with upcoming bills of {bills_str} accounted for.{goal_note}"
            )
            aria_priority = "polite"
        else:
            answer = (
                f"No, this purchase{item_str} of {amt_str} is not currently affordable. "
                f"After accounting for upcoming unpaid bills of {bills_str}, your available funds are insufficient."
            )
            aria_priority = "assertive"

        return {
            "intent": "check_affordability",
            "answer_text": answer,
            "aria_priority": aria_priority,
            "requires_confirmation": False,
            "confirmation_token": None,
            "pending_payment_id": None,
            "structured_facts": facts,
            "structured_data": facts,
            "conversation_status": "completed",
        }

    # =========================================================================
    # project_goal_completion
    # =========================================================================
    elif intent_name == "project_goal_completion":
        goal_name = facts.get("goal_name", "Savings Goal")
        months = facts.get("current_months_remaining", 0)
        monthly_str = _format_currency(facts.get("monthly_contribution", "0.00"))
        target_str = _format_currency(facts.get("target_amount", "0.00"))
        current_str = _format_currency(facts.get("current_amount", "0.00"))

        hypo_months = facts.get("hypothetical_months_remaining")
        hypo_note = f" With your proposed contribution, you would reach it in {hypo_months} month(s)." if hypo_months is not None else ""

        answer = (
            f"For your {goal_name} goal ({current_str} saved of {target_str}), "
            f"you are projected to reach completion in {months} month(s) at {monthly_str} per month.{hypo_note}"
        )
        return {
            "intent": "project_goal_completion",
            "answer_text": answer,
            "aria_priority": "polite",
            "requires_confirmation": False,
            "confirmation_token": None,
            "pending_payment_id": None,
            "structured_facts": facts,
            "structured_data": facts,
            "conversation_status": "completed",
        }

    # =========================================================================
    # get_insights
    # =========================================================================
    elif intent_name == "get_insights":
        insights_list = facts.get("insights", [])
        if not insights_list:
            answer = "You have no active spending alerts or anomalies at this time. Your finances look healthy."
        else:
            lines = [f"You have {len(insights_list)} financial insight(s):"]
            for ins in insights_list:
                i_type = ins.get("type")
                if i_type == "spending_increase":
                    lines.append(f"• Spending on {ins.get('category')} increased by {ins.get('pct')}% this month.")
                elif i_type == "subscription_increase":
                    lines.append(f"• Subscription for {ins.get('merchant')} ({ins.get('category')}) increased by {ins.get('pct')}%.")
                elif i_type == "upcoming_bill":
                    amt_b = _format_currency(ins.get("amount"))
                    lines.append(f"• Upcoming bill of {amt_b} for {ins.get('category')} due within 7 days.")
                else:
                    lines.append(f"• {i_type.replace('_', ' ').title()}.")
            answer = " ".join(lines)

        return {
            "intent": "get_insights",
            "answer_text": answer,
            "aria_priority": "polite",
            "requires_confirmation": False,
            "confirmation_token": None,
            "pending_payment_id": None,
            "structured_facts": facts,
            "structured_data": facts,
            "conversation_status": "completed",
        }

    # =========================================================================
    # payment_preview
    # =========================================================================
    elif intent_name == "payment_preview":
        amt_str = _format_currency(facts.get("amount", "0.00"))
        recipient = facts.get("recipient_name", "the recipient")
        bal_after_str = _format_currency(facts.get("balance_after", "0.00"))
        bills_str = _format_currency(facts.get("upcoming_bills", "0.00"))
        fraud_warning = facts.get("fraud_warning", False)
        risk_reasons = facts.get("risk_reasons", [])
        pending_id = facts.get("pending_payment_id")

        warning_prefix = ""
        aria_priority = "polite"
        if fraud_warning:
            aria_priority = "assertive"
            warning_prefix = "Security Warning: " + " ".join(risk_reasons) + " "

        answer = (
            f"{warning_prefix}You are about to send {amt_str} to {recipient}. "
            f"Your balance after payment will be {bal_after_str}, leaving upcoming bills of {bills_str} covered. "
            f"Please say 'confirm' or select confirm to execute this payment."
        )

        return {
            "intent": "payment_preview",
            "answer_text": answer,
            "aria_priority": aria_priority,
            "requires_confirmation": True,
            "confirmation_token": str(pending_id) if pending_id else None,
            "pending_payment_id": pending_id,
            "structured_facts": facts,
            "structured_data": facts,
            "conversation_status": "awaiting_confirmation",
        }

    # =========================================================================
    # payment_execute
    # =========================================================================
    elif intent_name == "payment_execute":
        amt_str = _format_currency(facts.get("amount", "0.00"))
        recipient = facts.get("recipient_name", "the recipient")
        new_bal_str = _format_currency(facts.get("new_balance", "0.00"))
        tx_id = facts.get("transaction_id")

        answer = (
            f"Payment of {amt_str} to {recipient} was successfully completed. "
            f"Your updated authoritative balance is {new_bal_str} (Transaction #{tx_id})."
        )

        return {
            "intent": "payment_execute",
            "answer_text": answer,
            "aria_priority": "polite",
            "requires_confirmation": False,
            "confirmation_token": None,
            "pending_payment_id": facts.get("pending_payment_id"),
            "structured_facts": facts,
            "structured_data": facts,
            "conversation_status": "completed",
        }

    return {
        "intent": intent_name,
        "answer_text": "Request processed successfully.",
        "aria_priority": "polite",
        "requires_confirmation": False,
        "confirmation_token": None,
        "pending_payment_id": None,
        "structured_facts": facts,
        "structured_data": facts,
        "conversation_status": "completed",
    }


def explain(
    intent_data: Dict[str, Any],
    facts: Dict[str, Any],
    query: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    preferred_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synthesizes natural language narration and ARIA priority from structured backend facts.
    Tries REAL_LLM grounded explanation first if requested and available, falling back
    to deterministic template narration.
    """
    client = llm_client or LLMClient()
    intent_name = intent_data.get("intent", facts.get("intent", "unknown"))

    # Determine desired execution mode
    mode = preferred_mode or intent_data.get("execution_mode", "MOCK_FALLBACK")

    # If status is clarification_needed or unsupported, template gives clearest deterministic response
    if facts.get("status") in ("clarification_needed", "unsupported_intent"):
        res = explain_fallback(intent_data=intent_data, facts=facts, query=query)
        res["execution_mode"] = mode
        return res

    if mode == "REAL_LLM" and client.is_available():
        llm_text, aria_prio, err = client.explain_facts(intent=intent_name, facts=facts, query=query)
        if llm_text:
            pending_id = facts.get("pending_payment_id")
            req_confirm = bool(facts.get("requires_confirmation", False))
            conv_status = "awaiting_confirmation" if req_confirm else "completed"

            return {
                "intent": intent_name,
                "answer_text": llm_text,
                "aria_priority": aria_prio or "polite",
                "requires_confirmation": req_confirm,
                "confirmation_token": str(pending_id) if pending_id else None,
                "pending_payment_id": pending_id,
                "structured_facts": facts,
                "structured_data": facts,
                "execution_mode": "REAL_LLM",
                "conversation_status": conv_status,
            }
        else:
            logger.warning(f"LLM Explainer failed, degrading to MOCK_FALLBACK: {err}")

    # Fallback to deterministic template explainer
    res = explain_fallback(intent_data=intent_data, facts=facts, query=query)
    res["execution_mode"] = "MOCK_FALLBACK"
    return res
