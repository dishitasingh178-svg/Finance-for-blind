"""
Integration Tests for FinSight AI Conversational Router (POST /ask and POST /api/v1/ask).

Tests:
- Balance queries (direct and natural-language / conversational)
- Categorical spending queries
- Affordability evaluation with natural-language amounts ("8k", "50 thousand")
- Missing information clarification loops (missing amount, missing goal)
- Goal progress and completion projection queries
- Off-topic / non-financial queries
- Non-existent user validation (404)
- Empty query validation (422)
- Versioned endpoint parity (/api/v1/ask)
- Live database integration with real seeded demo user
"""

from decimal import Decimal
from datetime import datetime, date
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import User, Account, Transaction, Goal, Bill


@pytest.fixture
def ai_user_fixture(db_session: Session) -> dict:
    """Sets up a complete user profile with transactions, bills, and goals."""
    user = User(
        full_name="Aarav Sharma",
        email="aarav.ai@example.com",
        accessibility_prefs={
            "voice_first": True,
            "screen_reader": True,
            "spoken_confirmations": True,
            "preferred_language": "en-IN",
        },
    )
    db_session.add(user)
    db_session.flush()

    account = Account(
        user_id=user.id,
        name="Primary Savings",
        account_type="savings",
        balance=Decimal("138372.00"),
        monthly_income=Decimal("75000.00"),
        currency="INR",
        is_active=True,
    )
    db_session.add(account)
    db_session.flush()

    # Transactions
    txs = [
        Transaction(
            account_id=account.id,
            user_id=user.id,
            amount=Decimal("150000.00"),
            transaction_type="income",
            category="Other",
            description="Salary & Opening",
            transaction_date=datetime(2026, 8, 1, 9, 0),
        ),
        Transaction(
            account_id=account.id,
            user_id=user.id,
            amount=Decimal("-14450.00"),
            transaction_type="expense",
            category="Food",
            merchant_name="BigBasket",
            description="August Groceries",
            transaction_date=datetime(2026, 8, 20, 18, 0),
        ),
        Transaction(
            account_id=account.id,
            user_id=user.id,
            amount=Decimal("-11850.00"),
            transaction_type="expense",
            category="Food",
            merchant_name="BigBasket",
            description="July Groceries",
            transaction_date=datetime(2026, 7, 15, 12, 0),
        ),
    ]
    db_session.add_all(txs)

    # Active Goal: Emergency Fund
    goal = Goal(
        user_id=user.id,
        name="Emergency Fund",
        target_amount=Decimal("150000.00"),
        current_amount=Decimal("45000.00"),
        monthly_contribution=Decimal("10000.00"),
        currency="INR",
        target_date=date(2027, 6, 30),
        status="active",
    )
    db_session.add(goal)

    # Upcoming unpaid bills (total: 6529.00)
    bills = [
        Bill(
            user_id=user.id,
            name="Electricity Bill",
            amount=Decimal("1850.00"),
            currency="INR",
            category="Bills",
            due_date=date(2026, 9, 5),
            status="unpaid",
        ),
        Bill(
            user_id=user.id,
            name="Broadband",
            amount=Decimal("1179.00"),
            currency="INR",
            category="Bills",
            due_date=date(2026, 9, 10),
            status="unpaid",
        ),
        Bill(
            user_id=user.id,
            name="Maintenance",
            amount=Decimal("3500.00"),
            currency="INR",
            category="Bills",
            due_date=date(2026, 9, 15),
            status="unpaid",
        ),
    ]
    db_session.add_all(bills)
    db_session.commit()

    return {
        "user_id": user.id,
        "account_id": account.id,
        "goal_id": goal.id,
    }


class TestAIConversationalRouter:
    """Test suite for POST /ask and POST /api/v1/ask."""

    def test_ask_balance_query(self, client: TestClient, ai_user_fixture: dict):
        user_id = ai_user_fixture["user_id"]
        response = client.post(
            "/ask",
            json={"user_id": user_id, "query": "What's my balance?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer_text" in data
        assert "structured_data" in data
        assert "123,700" in data["answer_text"] or "123700" in data["answer_text"]
        assert Decimal(str(data["structured_data"]["balance"])) == Decimal("123700.00")

    def test_ask_conversational_balance_query(self, client: TestClient, ai_user_fixture: dict):
        user_id = ai_user_fixture["user_id"]
        response = client.post(
            "/ask",
            json={"user_id": user_id, "query": "how much money do i have left in my account?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "123,700" in data["answer_text"] or "123700" in data["answer_text"]
        assert Decimal(str(data["structured_data"]["balance"])) == Decimal("123700.00")

    def test_ask_spending_summary_food(self, client: TestClient, ai_user_fixture: dict):
        user_id = ai_user_fixture["user_id"]
        response = client.post(
            "/ask",
            json={"user_id": user_id, "query": "how much did I spend on food this month?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "14,450" in data["answer_text"] or "14450" in data["answer_text"]
        assert Decimal(str(data["structured_data"]["by_category"]["Food"])) == Decimal("14450.00")

    def test_ask_affordability_with_8k(self, client: TestClient, ai_user_fixture: dict):
        user_id = ai_user_fixture["user_id"]
        response = client.post(
            "/ask",
            json={"user_id": user_id, "query": "Can I afford headphones for 8k?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["structured_data"]["can_afford"] is True
        assert Decimal(str(data["structured_data"]["upcoming_bills"])) == Decimal("6529.00")
        assert "afford" in data["answer_text"].lower()

    def test_ask_affordability_with_fifty_thousand(self, client: TestClient, ai_user_fixture: dict):
        user_id = ai_user_fixture["user_id"]
        response = client.post(
            "/ask",
            json={"user_id": user_id, "query": "Can I buy a laptop for 50 thousand?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["structured_data"]["can_afford"] is True
        assert Decimal(str(data["structured_data"]["balance_after"])) == Decimal("73700.00")

    def test_ask_missing_affordability_amount_clarification(self, client: TestClient, ai_user_fixture: dict):
        user_id = ai_user_fixture["user_id"]
        response = client.post(
            "/ask",
            json={"user_id": user_id, "query": "Can I afford it?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "cost" in data["answer_text"].lower() or "price" in data["answer_text"].lower() or "much" in data["answer_text"].lower()
        assert data["structured_data"].get("status") == "clarification_needed"

    def test_ask_goal_projection_query(self, client: TestClient, ai_user_fixture: dict):
        user_id = ai_user_fixture["user_id"]
        response = client.post(
            "/ask",
            json={"user_id": user_id, "query": "When will I reach my emergency fund?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert Decimal(str(data["structured_data"]["current_months_remaining"])) == Decimal("11")
        assert "11" in data["answer_text"]

    def test_ask_missing_goal_clarification(self, client: TestClient, ai_user_fixture: dict):
        user_id = ai_user_fixture["user_id"]
        response = client.post(
            "/ask",
            json={"user_id": user_id, "query": "how much longer do I need to save?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "goal" in data["answer_text"].lower()
        assert data["structured_data"].get("status") == "clarification_needed"

    def test_ask_nonexistent_user_returns_404(self, client: TestClient):
        response = client.post(
            "/ask",
            json={"user_id": 99999, "query": "What's my balance?"},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_ask_empty_query_rejected_422(self, client: TestClient, ai_user_fixture: dict):
        user_id = ai_user_fixture["user_id"]
        response = client.post(
            "/ask",
            json={"user_id": user_id, "query": ""},
        )
        assert response.status_code == 422

    def test_ask_off_topic_query(self, client: TestClient, ai_user_fixture: dict):
        user_id = ai_user_fixture["user_id"]
        response = client.post(
            "/ask",
            json={"user_id": user_id, "query": "Tell me something unrelated like the weather in Mumbai."},
        )
        assert response.status_code == 200
        data = response.json()
        assert "finance" in data["answer_text"].lower() or "finsight" in data["answer_text"].lower() or "clarify" in data["answer_text"].lower()

    def test_versioned_endpoint_api_v1_ask(self, client: TestClient, ai_user_fixture: dict):
        user_id = ai_user_fixture["user_id"]
        response = client.post(
            "/api/v1/ask",
            json={"user_id": user_id, "query": "What's my balance?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer_text" in data
        assert "123,700" in data["answer_text"] or "123700" in data["answer_text"]


class TestLiveSeededAIDataset:
    """Verifies AI endpoint against the live seeded database."""

    def test_live_seeded_user_ask(self):
        from backend.main import app
        with TestClient(app) as live_client:
            response = live_client.post(
                "/ask",
                json={"user_id": 1, "query": "What's my balance?"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "138,372" in data["answer_text"] or "138372" in data["answer_text"]
            assert Decimal(str(data["structured_data"]["balance"])) == Decimal("138372.00")

