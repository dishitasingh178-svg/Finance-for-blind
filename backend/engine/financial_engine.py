"""
Deterministic Financial Engine for FinSight.

ARCHITECTURAL PRINCIPLES:
-------------------------
1. Absolute Separation of Math and Language:
   - The LLM MUST NEVER perform financial calculations.
   - The LLM MUST NEVER query the database directly.
   - All computation is performed deterministically using Decimal precision.
2. Authoritative Balance:
   - Authoritative balance is defined strictly as SUM(transaction.amount) across all user accounts.
   - accounts.balance is a cached/display value and is NEVER used by the financial engine.
3. Money Sign Convention:
   - Positive (+) = Money entering account (Income, Opening Balance, Refund).
   - Negative (-) = Money leaving account (Expenses, Purchases, Bills).
4. Plain Functions:
   - No service classes. Public API is exposed as plain typed functions.

NOTE: This file contains foundation stubs. Calculations will be implemented in Day 2.
"""

from decimal import Decimal
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session


def get_balance(user_id: int, db: Session) -> Dict[str, Any]:
    """
    Calculates the authoritative user balance from transaction history.

    Authoritative balance = SUM(transaction.amount) for all transactions
    belonging to the user's accounts.

    Args:
        user_id: The ID of the user.
        db: SQLAlchemy database session.

    Returns:
        Dict containing structured balance facts:
        {
            "user_id": int,
            "authoritative_balance": Decimal,
            "currency": str,
            "account_breakdown": List[Dict[str, Any]]
        }
    """
    raise NotImplementedError("Financial calculations will be implemented in Day 2.")


def get_spending_summary(user_id: int, db: Session, period: str = "this_month") -> Dict[str, Any]:
    """
    Computes deterministic category spending totals and income/expense breakdown
    for the specified time period.

    Args:
        user_id: The ID of the user.
        db: SQLAlchemy database session.
        period: Time period filter ("this_month", "last_month", "last_30_days", etc.)

    Returns:
        Dict containing structured spending breakdown facts.
    """
    raise NotImplementedError("Financial calculations will be implemented in Day 2.")


def check_affordability(user_id: int, amount: Decimal, db: Session) -> Dict[str, Any]:
    """
    Deterministically evaluates whether a user can afford an expense of `amount`,
    taking into account authoritative balance, upcoming unpaid bills, and recurring commitments.

    Args:
        user_id: The ID of the user.
        amount: The purchase amount to evaluate (positive Decimal).
        db: SQLAlchemy database session.

    Returns:
        Dict containing structured affordability facts:
        {
            "affordable": bool,
            "risk_level": str ("LOW", "MODERATE", "HIGH"),
            "authoritative_balance": Decimal,
            "upcoming_unpaid_bills_total": Decimal,
            "discretionary_funds": Decimal,
            "projected_remaining": Decimal
        }
    """
    raise NotImplementedError("Financial calculations will be implemented in Day 2.")


def project_goal_completion(
    goal_id: int,
    db: Session,
    hypothetical_contribution: Optional[Decimal] = None
) -> Dict[str, Any]:
    """
    Calculates deterministic goal projection and estimated completion timeline
    using exact decimal math.

    Args:
        goal_id: The ID of the goal.
        db: SQLAlchemy database session.
        hypothetical_contribution: Optional override for monthly contribution simulation.

    Returns:
        Dict containing structured projection facts.
    """
    raise NotImplementedError("Financial calculations will be implemented in Day 2.")


def get_insights(user_id: int, db: Session) -> List[Dict[str, Any]]:
    """
    Generates a list of deterministic structured financial insights
    (e.g., spending anomalies, upcoming bill alerts, goal progress milestones).

    Returns structured facts only — zero natural language narration or TTS formatting.

    Args:
        user_id: The ID of the user.
        db: SQLAlchemy database session.

    Returns:
        List of structured insight dicts.
    """
    raise NotImplementedError("Financial calculations will be implemented in Day 2.")
