"""
Transactions Router for FinSight.

Provides transaction history and deterministic category spending breakdowns for the specified period.
"""

from typing import List, Dict
from decimal import Decimal
from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import extract

from backend.db import get_db
from backend.models import User, Account, Transaction
from backend.engine import get_balance, get_spending_summary
from backend.engine.financial_engine import _get_previous_calendar_month
from backend.schemas import TransactionsListResponse, TransactionResponse

router = APIRouter(tags=["Transactions"])


@router.get("/transactions", response_model=TransactionsListResponse, summary="Get User Transactions")
@router.get("/api/v1/transactions", response_model=TransactionsListResponse, include_in_schema=False)
def get_user_transactions(
    user_id: int = Query(..., description="ID of the user"),
    period: str = Query("this_month", description="Period to filter ('this_month' or 'last_month')"),
    db: Session = Depends(get_db),
) -> TransactionsListResponse:
    """
    Returns transaction history for the specified user and period, along with deterministic
    category spending totals.

    Rules:
    - User ownership: Only transactions belonging to user-owned accounts are retrieved.
    - Period alignment: 'this_month' and 'last_month' use centralized calendar bounds matching the engine.
    - Sign convention: Inflows remain positive (+), outflows remain negative (-).
    - Categorical totals: Directly sourced from get_spending_summary(period=period).
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id {user_id} not found.",
        )

    if period not in ("this_month", "last_month"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported period '{period}'. Supported periods are: 'this_month', 'last_month'.",
        )

    # 1. Obtain categorical totals directly from the deterministic financial engine
    spending_summary = get_spending_summary(user_id, db, period=period)
    by_category: Dict[str, Decimal] = spending_summary["by_category"]

    # 2. Determine calendar period range matching the financial engine
    balance_info = get_balance(user_id, db)
    as_of = balance_info["as_of"]

    if not as_of:
        return TransactionsListResponse(transactions=[], by_category=by_category)

    if period == "this_month":
        target_year, target_month = as_of.year, as_of.month
    else:  # "last_month"
        target_year, target_month = _get_previous_calendar_month(as_of.year, as_of.month)

    # 3. Query transactions with strict user ownership and period filtering
    transactions = (
        db.query(Transaction)
        .join(Account, Transaction.account_id == Account.id)
        .filter(
            Account.user_id == user_id,
            Transaction.user_id == user_id,
            extract("year", Transaction.transaction_date) == target_year,
            extract("month", Transaction.transaction_date) == target_month,
        )
        .order_by(Transaction.transaction_date.asc())
        .all()
    )

    transactions_responses = [TransactionResponse.model_validate(t) for t in transactions]

    return TransactionsListResponse(
        transactions=transactions_responses,
        by_category=by_category,
    )
