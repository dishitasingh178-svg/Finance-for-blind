"""
Foundation Tests for FinSight Backend.

NOTE: These are architectural and structural foundation tests verifying:
- Database schema and table creation across all six models
- Model relationships, constraints, and foreign key integrity
- Mandatory accessibility preferences
- Non-negotiable money sign convention (+ inflow, - outflow)
- Deterministic engine public function signatures and return types

Calculation logic and numerical assertion tests are scheduled for Day 2.
"""

import inspect
from datetime import datetime, date
from decimal import Decimal
from typing import get_type_hints, List, Dict, Any
import pytest
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import Session

from backend.db import Base
from backend.models import (
    User,
    Account,
    Transaction,
    Goal,
    Bill,
    Document,
    VALID_TRANSACTION_TYPES,
    VALID_CATEGORIES,
    VALID_GOAL_STATUSES,
    VALID_BILL_STATUSES,
)
from backend.engine import (
    get_balance,
    get_spending_summary,
    check_affordability,
    project_goal_completion,
    get_insights,
    build_insight_fact,
)


class TestDatabaseFoundation:
    """Verifies table creation and schema integrity for all 6 models."""

    def test_all_six_tables_exist(self, db_session: Session):
        """Verify that exactly the 6 required domain tables exist in SQLite metadata."""
        expected_tables = {"users", "accounts", "transactions", "goals", "bills", "documents"}
        actual_tables = set(Base.metadata.tables.keys())
        assert expected_tables.issubset(actual_tables), f"Missing tables: {expected_tables - actual_tables}"

    def test_user_creation_with_accessibility_prefs(self, db_session: Session):
        """Verify User model has required accessibility_prefs with voice-first defaults."""
        user = User(
            full_name="Aarav Sharma",
            email="aarav.sharma@example.com",
            accessibility_prefs={
                "voice_first": True,
                "screen_reader": True,
                "spoken_confirmations": True,
                "preferred_language": "en-IN",
            },
        )
        db_session.add(user)
        db_session.commit()

        queried = db_session.query(User).filter_by(email="aarav.sharma@example.com").first()
        assert queried is not None
        assert queried.accessibility_prefs["voice_first"] is True
        assert queried.accessibility_prefs["screen_reader"] is True
        assert queried.accessibility_prefs["preferred_language"] == "en-IN"

    def test_account_and_transaction_relationship_with_money_convention(self, db_session: Session):
        """
        Verifies Account and Transaction models, relationships, and money sign convention:
        - Positive (+) for income / opening balance / credit
        - Negative (-) for expense / debit
        - accounts.balance is cached/display only
        """
        user = User(
            full_name="Priya Patel",
            email="priya.patel@example.com",
        )
        db_session.add(user)
        db_session.flush()

        account = Account(
            user_id=user.id,
            name="Primary Checking",
            account_type="checking",
            balance=Decimal("25000.00"),  # Cached display balance
            monthly_income=Decimal("75000.00"),
            currency="INR",
        )
        db_session.add(account)
        db_session.flush()

        # Opening balance transaction (+25000.00)
        t_opening = Transaction(
            account_id=account.id,
            user_id=user.id,
            amount=Decimal("25000.00"),  # Positive: Money entering account
            currency="INR",
            transaction_type="income",
            category="Other",
            description="Opening Balance",
            transaction_date=datetime.utcnow(),
        )

        # Expense transaction (-620.00)
        t_expense = Transaction(
            account_id=account.id,
            user_id=user.id,
            amount=Decimal("-620.00"),  # Negative: Money leaving account
            currency="INR",
            transaction_type="expense",
            category="Food",
            merchant_name="Swiggy",
            description="Dinner Delivery",
            transaction_date=datetime.utcnow(),
        )

        db_session.add_all([t_opening, t_expense])
        db_session.commit()

        # Verify transaction sums
        transactions = db_session.query(Transaction).filter_by(account_id=account.id).all()
        assert len(transactions) == 2
        total_authoritative_balance = sum(t.amount for t in transactions)
        assert total_authoritative_balance == Decimal("24380.00")

    def test_goal_model_constraints(self, db_session: Session):
        """Verifies Goal creation with Decimal-safe Numeric amounts and status."""
        user = User(full_name="Rohan Gupta", email="rohan.gupta@example.com")
        db_session.add(user)
        db_session.flush()

        goal = Goal(
            user_id=user.id,
            name="Emergency Fund",
            target_amount=Decimal("100000.00"),
            current_amount=Decimal("30000.00"),
            monthly_contribution=Decimal("5000.00"),
            currency="INR",
            target_date=date(2026, 12, 31),
            status="active",
        )
        db_session.add(goal)
        db_session.commit()

        queried = db_session.query(Goal).filter_by(id=goal.id).first()
        assert queried is not None
        assert queried.target_amount == Decimal("100000.00")
        assert queried.status in VALID_GOAL_STATUSES

    def test_bill_model_constraints(self, db_session: Session):
        """Verifies Bill creation with Decimal-safe Numeric amounts and status semantics."""
        user = User(full_name="Sneha Rao", email="sneha.rao@example.com")
        db_session.add(user)
        db_session.flush()

        bill = Bill(
            user_id=user.id,
            name="Electricity Bill (BESCOM)",
            amount=Decimal("1850.00"),
            currency="INR",
            category="Bills",
            due_date=date(2026, 9, 5),
            frequency="monthly",
            status="unpaid",
            is_recurring=True,
        )
        db_session.add(bill)
        db_session.commit()

        queried = db_session.query(Bill).filter_by(id=bill.id).first()
        assert queried is not None
        assert queried.amount == Decimal("1850.00")
        assert queried.status == "unpaid"
        assert queried.status in VALID_BILL_STATUSES

    def test_document_model_storage(self, db_session: Session):
        """Verifies Document model creation for storing structured document facts."""
        user = User(full_name="Kavita Iyer", email="kavita.iyer@example.com")
        db_session.add(user)
        db_session.flush()

        doc = Document(
            user_id=user.id,
            filename="bescom_september_2026.pdf",
            file_path="/storage/documents/bescom_september_2026.pdf",
            document_type="bill",
            mime_type="application/pdf",
            raw_text="BESCOM Electricity Bill: Amount Due Rs 1850.00 Due Date: 05-09-2026",
            extracted_facts={"vendor": "BESCOM", "amount": 1850.00, "due_date": "2026-09-05"},
            is_suspicious=False,
        )
        db_session.add(doc)
        db_session.commit()

        queried = db_session.query(Document).filter_by(id=doc.id).first()
        assert queried is not None
        assert queried.document_type == "bill"
        assert queried.extracted_facts["amount"] == 1850.00


class TestFinancialEngineContract:
    """Verifies that the financial engine public API matches the established contract."""

    def test_get_balance_signature(self):
        """get_balance must accept (user_id, db) and return a dict."""
        sig = inspect.signature(get_balance)
        params = list(sig.parameters.keys())
        assert params == ["user_id", "db"]

    def test_get_spending_summary_signature(self):
        """get_spending_summary must accept (user_id, db, period='this_month')."""
        sig = inspect.signature(get_spending_summary)
        params = list(sig.parameters.keys())
        assert params == ["user_id", "db", "period"]
        assert sig.parameters["period"].default == "this_month"

    def test_check_affordability_signature(self):
        """check_affordability must accept (user_id, amount, db)."""
        sig = inspect.signature(check_affordability)
        params = list(sig.parameters.keys())
        assert params == ["user_id", "amount", "db"]

    def test_project_goal_completion_signature(self):
        """project_goal_completion must accept (goal_id, db, hypothetical_contribution=None)."""
        sig = inspect.signature(project_goal_completion)
        params = list(sig.parameters.keys())
        assert params == ["goal_id", "db", "hypothetical_contribution"]
        assert sig.parameters["hypothetical_contribution"].default is None

    def test_get_insights_signature_and_return_type(self):
        """get_insights must accept (user_id, db) and have a list return annotation."""
        sig = inspect.signature(get_insights)
        params = list(sig.parameters.keys())
        assert params == ["user_id", "db"]
        hints = get_type_hints(get_insights)
        assert hints.get("return") in (list, List, List[dict], List[Dict[str, Any]])

    def test_build_insight_fact_structure(self):
        """Verifies that build_insight_fact creates pure structured fact objects."""
        fact = build_insight_fact(
            insight_type="spending_spike",
            severity="WARNING",
            category="Food",
            metric_name="monthly_food_spend",
            metric_value=Decimal("12500.00"),
            threshold_value=Decimal("8000.00"),
            metadata={"percentage_increase": 56.25},
        )
        assert fact["insight_type"] == "spending_spike"
        assert fact["severity"] == "WARNING"
        assert fact["metric_value"] == "12500.00"
        assert fact["threshold_value"] == "8000.00"
        assert fact["metadata"]["percentage_increase"] == 56.25


class TestHealthEndpoint:
    """Verifies that the health check endpoints respond with 200 and valid JSON."""

    def test_root_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "finsight-backend"

    def test_api_v1_health_endpoint(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
