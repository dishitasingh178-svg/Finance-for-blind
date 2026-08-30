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
    source: str = "bank"
    reference_id: Optional[str] = None
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


# --- Day 4B Ingestion Schemas ---

class VoiceTransactionRequest(BaseModel):
    """Request payload from voice/AI layer to ingest a structured transaction."""
    user_id: int = Field(..., description="ID of the user")
    account_id: Optional[int] = Field(None, description="Optional account ID (auto-assigns to active account if None)")
    amount: Decimal = Field(..., gt=0, description="Positive transaction amount")
    transaction_type: str = Field(..., description="'expense' or 'income'")
    category: str = Field("Other", description="Transaction category (Food, Transport, Shopping, Bills, etc.)")
    merchant_name: Optional[str] = Field(None, description="Payee / Merchant name")
    description: Optional[str] = Field(None, description="Transaction description or voice note")
    transaction_date: Optional[datetime] = Field(None, description="Transaction date/time (defaults to now)")


class BankConnectRequest(BaseModel):
    """Request payload to connect a user's account to a mock financial institution."""
    user_id: int = Field(..., description="ID of the user")
    institution_name: str = Field("HDFC Bank Mock", description="Mock bank institution name")
    account_id: Optional[int] = Field(None, description="Optional account ID to link")


class BankConnectResponse(BaseModel):
    """Response model for mock bank connection."""
    status: str
    institution_name: str
    user_id: int
    account_id: int
    message: str


class SkippedTransactionItem(BaseModel):
    """Details of a skipped duplicate transaction."""
    reference_id: Optional[str] = None
    merchant_name: Optional[str] = None
    amount: str
    reason: Optional[str] = None
    existing_transaction_id: Optional[int] = None


class BankSyncRequest(BaseModel):
    """Request payload to trigger mock bank feed synchronization."""
    user_id: int = Field(..., description="ID of the user")
    account_id: Optional[int] = Field(None, description="Optional account ID to sync")


class BankSyncResponse(BaseModel):
    """Response model for bank feed synchronization."""
    status: str
    user_id: int
    account_id: int
    imported_count: int
    duplicate_count: int
    skipped_count: int
    imported_transactions: List[TransactionResponse]
    skipped_transactions: List[SkippedTransactionItem]


class StatementCandidateItem(BaseModel):
    """Single extracted statement transaction candidate."""
    reference_id: Optional[str] = Field(None, description="Unique reference ID extracted from statement")
    amount: Decimal = Field(..., gt=0, description="Positive transaction amount")
    transaction_type: str = Field("expense", description="'expense' or 'income'")
    category: str = Field("Other", description="Transaction category")
    merchant_name: Optional[str] = Field(None, description="Payee / Merchant name")
    description: Optional[str] = Field(None, description="Transaction description")
    transaction_date: datetime = Field(..., description="Extracted transaction timestamp")


class StatementEvaluatedCandidate(BaseModel):
    """Candidate transaction with duplicate evaluation status."""
    candidate_id: str
    reference_id: Optional[str] = None
    amount: Decimal
    transaction_type: str
    category: str
    merchant_name: Optional[str] = None
    description: Optional[str] = None
    transaction_date: datetime
    is_duplicate: bool = False
    duplicate_reason: Optional[str] = None


class StatementUploadRequest(BaseModel):
    """Request payload uploading extracted candidates from a statement."""
    user_id: int = Field(..., description="ID of the user")
    account_id: Optional[int] = Field(None, description="Target account ID")
    filename: str = Field(..., description="Original statement filename")
    extracted_candidates: List[StatementCandidateItem] = Field(default_factory=list, description="Extracted transaction candidate list")


class StatementUploadResponse(BaseModel):
    """Response model staging statement candidates for user confirmation."""
    document_id: int
    filename: str
    total_candidates: int
    valid_candidates_count: int
    duplicate_candidates_count: int
    candidates: List[StatementEvaluatedCandidate]


class ConfirmTransactionsRequest(BaseModel):
    """Request payload to confirm and persist validated statement candidates."""
    user_id: int = Field(..., description="ID of the user")
    account_id: Optional[int] = Field(None, description="Target account ID")
    document_id: Optional[int] = Field(None, description="Optional associated document ID")
    candidates: List[StatementCandidateItem] = Field(..., min_length=1, description="List of candidates to confirm and persist")


class ConfirmTransactionsResponse(BaseModel):
    """Response model confirming persisted transactions."""
    status: str
    confirmed_count: int
    skipped_duplicates_count: int
    transactions: List[TransactionResponse]
    skipped_items: List[SkippedTransactionItem]


# --- Conversational AI / Voice Schemas ---

class AskRequest(BaseModel):
    """Request payload for conversational AI financial assistant."""
    user_id: int = Field(..., description="ID of the user asking the question")
    query: str = Field(..., min_length=1, description="Natural language personal finance question")
    conversation_id: Optional[str] = Field(None, description="Optional conversation session ID for multi-turn dialogues")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional conversation or application context")


class AskResponse(BaseModel):
    """Response model containing grounded natural-language answer and structured engine facts."""
    answer_text: str = Field(..., description="Grounded natural-language spoken response")
    structured_data: Any = Field(..., description="Authoritative structured data returned by deterministic engine")
    conversation_id: Optional[str] = Field(None, description="Active conversation session ID")
    conversation_status: Optional[str] = Field("active", description="Current conversation state (active, awaiting_clarification, completed)")
    execution_mode: Optional[str] = Field(None, description="Safe runtime indicator (REAL_LLM or MOCK_FALLBACK)")


# --- PROTECT Pillar (Scam & Fraud Safety) Schemas ---

class ScamIndicator(BaseModel):
    """Specific suspicious pattern indicator detected in a message."""
    type: str = Field(..., description="Category of indicator (e.g. urgency, otp_request, fake_reward, suspicious_link, kyc_threat, impersonation)")
    evidence: str = Field(..., description="Exact quoted or paraphrased evidence from the message text")


class ScamCheckRequest(BaseModel):
    """Request payload for scam safety check."""
    message: str = Field(..., min_length=1, description="Message text or SMS to analyze for scam/fraud patterns")
    user_id: Optional[int] = Field(None, description="Optional user ID for context or logging")


class ScamCheckResponse(BaseModel):
    """Response model for pattern-based scam safety assessment."""
    risk_level: str = Field(..., description="Risk category: 'low', 'medium', or 'high'")
    looks_suspicious: bool = Field(..., description="Whether the message exhibits suspicious scam or fraud patterns")
    indicators: List[ScamIndicator] = Field(default_factory=list, description="List of detected scam indicators and grounded evidence")
    explanation: str = Field(..., description="Short explanation grounded strictly in the supplied message")
    recommended_actions: List[str] = Field(default_factory=list, description="Actionable safety guidance for the user")
    limitations: str = Field(..., description="Explicit disclaimer clarifying this is a pattern-based AI assessment, not a deterministic fraud verification system")
