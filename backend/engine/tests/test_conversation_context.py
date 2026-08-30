"""
Integration tests for FinSight Multi-Turn Conversational Context System.

Tests:
- Single-turn backward compatibility
- Multi-turn affordability clarification and parameter resolution (e.g. "headphones" -> "8k")
- Multi-turn natural language numbers ("fifty thousand", "8k")
- Multi-turn savings goal clarification and resolution ("How long to save?" -> "Emergency fund")
- Topic switching during pending clarification
- User isolation (prevent cross-user session tampering)
- Invalid conversation ID handling (404)
- Conversation TTL expiration behavior
"""

from datetime import datetime, timedelta, date
from decimal import Decimal
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import User, Account, Transaction, Goal, Bill, ConversationSession


@pytest.fixture
def multi_user_fixture(db_session: Session) -> dict:
    """Sets up two distinct users to verify cross-user isolation and multi-turn dialogues."""
    # User 1: Aarav
    user1 = User(
        full_name="Aarav Sharma",
        email="aarav.context@example.com",
        accessibility_prefs={"voice_first": True},
    )
    db_session.add(user1)
    db_session.flush()

    acc1 = Account(
        user_id=user1.id,
        name="Aarav Savings",
        account_type="savings",
        balance=Decimal("138372.00"),
        monthly_income=Decimal("75000.00"),
        currency="INR",
        is_active=True,
    )
    db_session.add(acc1)
    db_session.flush()

    txs1 = [
        Transaction(
            account_id=acc1.id,
            user_id=user1.id,
            amount=Decimal("150000.00"),
            transaction_type="income",
            category="Other",
            description="Salary",
            transaction_date=datetime(2026, 8, 1, 9, 0),
        ),
        Transaction(
            account_id=acc1.id,
            user_id=user1.id,
            amount=Decimal("-14450.00"),
            transaction_type="expense",
            category="Food",
            merchant_name="BigBasket",
            description="August Groceries",
            transaction_date=datetime(2026, 8, 20, 18, 0),
        ),
    ]
    db_session.add_all(txs1)

    goal1 = Goal(
        user_id=user1.id,
        name="Emergency Fund",
        target_amount=Decimal("150000.00"),
        current_amount=Decimal("45000.00"),
        monthly_contribution=Decimal("10000.00"),
        currency="INR",
        target_date=date(2027, 6, 30),
        status="active",
    )
    db_session.add(goal1)

    bills1 = [
        Bill(
            user_id=user1.id,
            name="Electricity",
            amount=Decimal("1850.00"),
            currency="INR",
            category="Bills",
            due_date=date(2026, 9, 5),
            status="unpaid",
        ),
        Bill(
            user_id=user1.id,
            name="Broadband",
            amount=Decimal("1179.00"),
            currency="INR",
            category="Bills",
            due_date=date(2026, 9, 10),
            status="unpaid",
        ),
        Bill(
            user_id=user1.id,
            name="Maintenance",
            amount=Decimal("3500.00"),
            currency="INR",
            category="Bills",
            due_date=date(2026, 9, 15),
            status="unpaid",
        ),
    ]
    db_session.add_all(bills1)

    # User 2: Priya (For isolation testing)
    user2 = User(
        full_name="Priya Patel",
        email="priya.patel@example.com",
        accessibility_prefs={"voice_first": True},
    )
    db_session.add(user2)
    db_session.flush()

    acc2 = Account(
        user_id=user2.id,
        name="Priya Checking",
        account_type="checking",
        balance=Decimal("25000.00"),
        monthly_income=Decimal("50000.00"),
        currency="INR",
        is_active=True,
    )
    db_session.add(acc2)
    db_session.flush()

    db_session.commit()

    return {
        "user1_id": user1.id,
        "user2_id": user2.id,
        "goal1_id": goal1.id,
    }


class TestMultiTurnConversationalContext:
    """Test suite for FinSight Multi-Turn Conversational Memory & Clarifications."""

    def test_single_turn_backward_compatibility(self, client: TestClient, multi_user_fixture: dict):
        """Verify that requests without conversation_id still work and return a new conversation_id."""
        user_id = multi_user_fixture["user1_id"]
        response = client.post(
            "/ask",
            json={"user_id": user_id, "query": "What's my balance?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer_text" in data
        assert data["conversation_id"] is not None
        assert data["conversation_id"].startswith("conv_")
        assert data["conversation_status"] == "completed"

    def test_multiturn_affordability_clarification_flow(self, client: TestClient, multi_user_fixture: dict):
        """
        Turn 1: 'Can I afford headphones?' -> asks 'How much do the headphones cost?'
        Turn 2: '8k' -> continues previous request, evaluates ₹8,000 for headphones against balance.
        """
        user_id = multi_user_fixture["user1_id"]

        # Turn 1: Affordability with missing amount
        turn1 = client.post(
            "/ask",
            json={"user_id": user_id, "query": "Can I afford headphones?"},
        )
        assert turn1.status_code == 200
        data1 = turn1.json()
        conv_id = data1["conversation_id"]
        assert conv_id is not None
        assert data1["conversation_status"] == "awaiting_clarification"
        assert "much" in data1["answer_text"].lower() or "cost" in data1["answer_text"].lower()

        # Turn 2: Follow-up response with price '8k'
        turn2 = client.post(
            "/ask",
            json={"user_id": user_id, "query": "8k", "conversation_id": conv_id},
        )
        assert turn2.status_code == 200
        data2 = turn2.json()
        assert data2["conversation_id"] == conv_id
        assert data2["conversation_status"] == "completed"
        assert data2["structured_data"]["can_afford"] is True
        assert Decimal(str(data2["structured_data"]["upcoming_bills"])) == Decimal("6529.00")
        assert Decimal(str(data2["structured_data"]["balance_after"])) == Decimal("127550.00")
        assert "afford" in data2["answer_text"].lower()

    def test_multiturn_affordability_with_word_amount(self, client: TestClient, multi_user_fixture: dict):
        """Turn 1: 'Can I buy a laptop?' -> Turn 2: 'fifty thousand'."""
        user_id = multi_user_fixture["user1_id"]

        turn1 = client.post(
            "/ask",
            json={"user_id": user_id, "query": "Can I buy a laptop?"},
        )
        assert turn1.status_code == 200
        data1 = turn1.json()
        conv_id = data1["conversation_id"]
        assert data1["conversation_status"] == "awaiting_clarification"

        turn2 = client.post(
            "/ask",
            json={"user_id": user_id, "query": "fifty thousand", "conversation_id": conv_id},
        )
        assert turn2.status_code == 200
        data2 = turn2.json()
        assert data2["conversation_status"] == "completed"
        assert data2["structured_data"]["can_afford"] is True
        assert Decimal(str(data2["structured_data"]["balance_after"])) == Decimal("85550.00")

    def test_multiturn_goal_clarification_flow(self, client: TestClient, multi_user_fixture: dict):
        """
        Turn 1: 'How much longer do I need to save?' -> asks 'Which savings goal would you like me to check?'
        Turn 2: 'My emergency fund' -> projects Emergency Fund completion (11 months).
        """
        user_id = multi_user_fixture["user1_id"]

        turn1 = client.post(
            "/ask",
            json={"user_id": user_id, "query": "How much longer do I need to save?"},
        )
        assert turn1.status_code == 200
        data1 = turn1.json()
        conv_id = data1["conversation_id"]
        assert data1["conversation_status"] == "awaiting_clarification"
        assert "goal" in data1["answer_text"].lower()

        turn2 = client.post(
            "/ask",
            json={"user_id": user_id, "query": "My emergency fund", "conversation_id": conv_id},
        )
        assert turn2.status_code == 200
        data2 = turn2.json()
        assert data2["conversation_status"] == "completed"
        assert Decimal(str(data2["structured_data"]["current_months_remaining"])) == Decimal("11")
        assert "11" in data2["answer_text"]

    def test_topic_switching_during_clarification(self, client: TestClient, multi_user_fixture: dict):
        """
        Turn 1: 'Can I afford headphones?' -> awaiting clarification.
        Turn 2: User changes topic to 'What's my balance?' -> cleanly executes balance inquiry.
        """
        user_id = multi_user_fixture["user1_id"]

        turn1 = client.post(
            "/ask",
            json={"user_id": user_id, "query": "Can I afford headphones?"},
        )
        assert turn1.status_code == 200
        conv_id = turn1.json()["conversation_id"]

        turn2 = client.post(
            "/ask",
            json={"user_id": user_id, "query": "What's my balance?", "conversation_id": conv_id},
        )
        assert turn2.status_code == 200
        data2 = turn2.json()
        assert data2["conversation_status"] == "completed"
        assert Decimal(str(data2["structured_data"]["balance"])) == Decimal("135550.00")

    def test_invalid_conversation_id_returns_404(self, client: TestClient, multi_user_fixture: dict):
        """Querying with a non-existent conversation_id returns 404."""
        user_id = multi_user_fixture["user1_id"]
        response = client.post(
            "/ask",
            json={"user_id": user_id, "query": "What's my balance?", "conversation_id": "conv_nonexistent_999"},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_cross_user_isolation(self, client: TestClient, multi_user_fixture: dict):
        """User 2 must NOT be able to access or mutate User 1's conversation session."""
        user1_id = multi_user_fixture["user1_id"]
        user2_id = multi_user_fixture["user2_id"]

        # Create session for User 1
        turn1 = client.post(
            "/ask",
            json={"user_id": user1_id, "query": "Can I afford headphones?"},
        )
        assert turn1.status_code == 200
        user1_conv_id = turn1.json()["conversation_id"]

        # User 2 attempts to use User 1's conversation_id
        tampered_response = client.post(
            "/ask",
            json={"user_id": user2_id, "query": "8k", "conversation_id": user1_conv_id},
        )
        assert tampered_response.status_code == 404
        assert "not found" in tampered_response.json()["detail"].lower()

    def test_conversation_ttl_expiration(self, client: TestClient, db_session: Session, multi_user_fixture: dict):
        """Verify that an expired conversation session safely resets state without crashing."""
        user_id = multi_user_fixture["user1_id"]

        turn1 = client.post(
            "/ask",
            json={"user_id": user_id, "query": "Can I afford headphones?"},
        )
        assert turn1.status_code == 200
        conv_id = turn1.json()["conversation_id"]

        # Simulate 20 minutes passed (past 15 minute TTL)
        session_record = db_session.query(ConversationSession).filter(ConversationSession.id == conv_id).first()
        assert session_record is not None
        session_record.updated_at = datetime.utcnow() - timedelta(minutes=20)
        db_session.commit()

        # Follow-up on expired session: should reset and process fresh query without crash
        turn2 = client.post(
            "/ask",
            json={"user_id": user_id, "query": "What's my balance?", "conversation_id": conv_id},
        )
        assert turn2.status_code == 200
        data2 = turn2.json()
        assert Decimal(str(data2["structured_data"]["balance"])) == Decimal("135550.00")
        assert data2["conversation_status"] == "completed"
