"""
Pydantic Schemas for FinSight API layer.

Defines typed request and response schemas with Decimal-safe precision and JSON serialization.
"""

from decimal import Decimal
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field


class GoalCreateRequest(BaseModel):
    """Request payload for creating a new financial goal."""
    user_id: int = Field(..., description="ID of the user who owns the goal")
    name: str = Field(..., min_length=1, max_length=255, description="Name of the financial goal")
    target_amount: Decimal = Field(..., gt=0, description="Target savings amount (must be positive)")
    monthly_contribution: Decimal = Field(..., gt=0, description="Monthly planned contribution (must be positive)")
    target_date: Optional[date] = Field(None, description="Optional target completion date")


class GoalUpdateRequest(BaseModel):
    """Request payload for updating a financial goal's contribution."""
    monthly_contribution: Decimal = Field(..., gt=0, description="Updated monthly contribution amount (must be positive)")
    user_id: Optional[int] = Field(None, description="Optional user_id for explicit ownership verification")


class GoalResponse(BaseModel):
    """Response model for a single financial goal."""
    id: int
    user_id: int
    name: str
    target_amount: Decimal
    current_amount: Decimal
    monthly_contribution: Decimal
    currency: str = "INR"
    target_date: Optional[date] = None
    status: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class GoalProjection(BaseModel):
    """Goal projection completion facts computed by the deterministic engine."""
    current_months_remaining: Decimal
    hypothetical_months_remaining: Optional[Decimal] = None


class GoalWithProjectionResponse(BaseModel):
    """Response model returning an updated goal along with its deterministic projection."""
    goal: GoalResponse
    projection: GoalProjection


class TransactionResponse(BaseModel):
    """Response model for a single financial transaction."""
    id: int
    account_id: int
    user_id: int
    amount: Decimal
    currency: str = "INR"
    transaction_type: str
    category: str
    merchant_name: Optional[str] = None
    description: Optional[str] = None
    transaction_date: datetime
    is_suspicious: bool = False

    model_config = ConfigDict(from_attributes=True)


class TransactionsListResponse(BaseModel):
    """Response model for transaction history and categorical spending breakdown."""
    transactions: List[TransactionResponse]
    by_category: Dict[str, Decimal]


class DashboardOverviewResponse(BaseModel):
    """
    Response model for user dashboard overview.

    Definition of terms:
    - balance: Authoritative balance derived from SUM(transaction.amount).
    - monthly_income: Sum of monthly_income across active accounts.
    - monthly_spending: Sourced from get_spending_summary(user_id, period='this_month')['total'].
    - monthly_surplus: Authoritative calculated cash-flow metric defined strictly as
      (monthly_income - monthly_spending) for the period.
    - savings: Legacy compatibility field equivalent to monthly_surplus (cash-flow surplus
      for the period, NOT confirmed deposits into a savings account).
    - upcoming_bills: Unpaid bills due within 30 days of the deterministic as_of date.
    - goals: List of active financial goals.
    """
    balance: Decimal = Field(..., description="Authoritative balance from transaction history")
    monthly_income: Decimal = Field(..., description="Total monthly income from active accounts")
    monthly_spending: Decimal = Field(..., description="Total expenses for the current month")
    monthly_surplus: Decimal = Field(..., description="Authoritative monthly cash-flow surplus (monthly_income - monthly_spending)")
    savings: Decimal = Field(..., description="Compatibility alias for monthly_surplus (cash-flow surplus, not savings-account deposits)")
    upcoming_bills: Decimal = Field(..., description="Total unpaid bills due within 30 days")
    goals: List[GoalResponse] = Field(..., description="List of active savings and financial goals")

