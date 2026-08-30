"""
Integration Tests for FinSight Backend UI Control Intents (POST /ask).

Tests:
1. sync_bank English triggers ("Sync my bank", "Refresh my account", "Update my bank")
2. sync_bank Hinglish triggers ("Bank update kar do", "Bank sync karo", "Mera bank update karo")
3. read_recent_transactions triggers ("Read my recent transactions", "Last transactions kya hai?", "What did I spend recently?")
4. read_goals triggers ("Read my goals", "Tell me my goals", "Mera goal progress kya hai?")
5. upload_document triggers ("Upload a document", "I want to upload my bank statement", "Bank statement scan karo")
6. Invariant: Financial queries still route to the financial engine
7. Invariant: Scam queries still route to check_scam_message
8. Invariant: UI-control intents NEVER invoke the financial engine
9. Frontend schema verification: structured_data contains frontend-detectable action & intent
10. Multi-turn conversation support with UI control commands
"""

from decimal import Decimal
from datetime import datetime
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import User, Account, Transaction
from backend.main import app
import backend.engine.financial_engine as real_engine


@pytest.fixture
def ui_user_fixture(db_session: Session) -> dict:
    """Sets up a complete user profile for testing conversational UI control commands."""
    user = User(
        full_name="Vikram Verma",
        email="vikram.ui@example.com",
        accessibility_prefs={
            "voice_first": True,
            "screen_reader": True,
        },
    )
    db_session.add(user)
    db_session.flush()

    account = Account(
        user_id=user.id,
        name="Primary Savings",
        account_type="savings",
        balance=Decimal("75000.00"),
        monthly_income=Decimal("50000.00"),
        currency="INR",
        is_active=True,
    )
    db_session.add(account)
    db_session.flush()

    tx = Transaction(
        account_id=account.id,
        user_id=user.id,
        amount=Decimal("75000.00"),
        transaction_type="income",
        category="Other",
        description="Salary",
        transaction_date=datetime(2026, 8, 1, 10, 0),
    )
    db_session.add(tx)
    db_session.commit()

    return {
        "user_id": user.id,
        "account_id": account.id,
    }


class TestUIControlIntents:
    """Test suite verifying the 4 UI Control Intents in the conversational pipeline."""

    @pytest.mark.parametrize("query", [
        "Sync my bank",
        "Refresh my account",
        "Update my bank",
    ])
    def test_sync_bank_english_triggers(self, client: TestClient, ui_user_fixture: dict, query: str, monkeypatch):
        """Test 1: sync_bank English triggers."""
        user_id = ui_user_fixture["user_id"]

        monkeypatch.setattr(
            "ai.pipeline.route_query",
            lambda *args, **kwargs: {"status": "success", "function_name": "sync_bank", "arguments": {}},
        )

        response = client.post(
            "/ask",
            json={"user_id": user_id, "query": query},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["conversation_status"] == "completed"
        assert "structured_data" in data
        assert data["structured_data"]["action"] == "sync_bank"
        assert data["structured_data"]["intent"] == "sync_bank"
        assert data["structured_data"]["status"] == "success"
        assert "sync" in data["answer_text"].lower() or "bank" in data["answer_text"].lower()

    @pytest.mark.parametrize("query", [
        "Bank update kar do",
        "Bank sync karo",
    ])
    def test_sync_bank_hinglish_triggers(self, client: TestClient, ui_user_fixture: dict, query: str, monkeypatch):
        """Test 2: sync_bank Hinglish triggers."""
        user_id = ui_user_fixture["user_id"]

        monkeypatch.setattr(
            "ai.pipeline.route_query",
            lambda *args, **kwargs: {"status": "success", "function_name": "sync_bank", "arguments": {}},
        )

        response = client.post(
            "/ask",
            json={"user_id": user_id, "query": query},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["structured_data"]["action"] == "sync_bank"
        assert data["structured_data"]["intent"] == "sync_bank"
        assert data["conversation_status"] == "completed"

    @pytest.mark.parametrize("query", [
        "Read my recent transactions",
        "Read my transactions",
        "Last transactions kya hai?",
        "What did I spend recently?",
    ])
    def test_read_recent_transactions_triggers(self, client: TestClient, ui_user_fixture: dict, query: str, monkeypatch):
        """Test 3: read_recent_transactions triggers."""
        user_id = ui_user_fixture["user_id"]

        monkeypatch.setattr(
            "ai.pipeline.route_query",
            lambda *args, **kwargs: {"status": "success", "function_name": "read_recent_transactions", "arguments": {}},
        )

        response = client.post(
            "/ask",
            json={"user_id": user_id, "query": query},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["conversation_status"] == "completed"
        assert data["structured_data"]["action"] == "read_recent_transactions"
        assert data["structured_data"]["intent"] == "read_recent_transactions"
        assert "transaction" in data["answer_text"].lower()

    @pytest.mark.parametrize("query", [
        "Read my goals",
        "Tell me my goals",
        "Mera goal progress kya hai?",
        "Read my goal progress",
    ])
    def test_read_goals_triggers(self, client: TestClient, ui_user_fixture: dict, query: str, monkeypatch):
        """Test 4: read_goals triggers."""
        user_id = ui_user_fixture["user_id"]

        monkeypatch.setattr(
            "ai.pipeline.route_query",
            lambda *args, **kwargs: {"status": "success", "function_name": "read_goals", "arguments": {}},
        )

        response = client.post(
            "/ask",
            json={"user_id": user_id, "query": query},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["conversation_status"] == "completed"
        assert data["structured_data"]["action"] == "read_goals"
        assert data["structured_data"]["intent"] == "read_goals"
        assert "goal" in data["answer_text"].lower()

    @pytest.mark.parametrize("query", [
        "Upload a document",
        "I want to upload my bank statement",
        "Bank statement scan karo",
        "Upload my statement",
    ])
    def test_upload_document_triggers(self, client: TestClient, ui_user_fixture: dict, query: str, monkeypatch):
        """Test 5: upload_document triggers."""
        user_id = ui_user_fixture["user_id"]

        monkeypatch.setattr(
            "ai.pipeline.route_query",
            lambda *args, **kwargs: {"status": "success", "function_name": "upload_document", "arguments": {}},
        )

        response = client.post(
            "/ask",
            json={"user_id": user_id, "query": query},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["conversation_status"] == "completed"
        assert data["structured_data"]["action"] == "upload_document"
        assert data["structured_data"]["intent"] == "upload_document"
        assert "document" in data["answer_text"].lower() or "upload" in data["answer_text"].lower() or "statement" in data["answer_text"].lower()

    def test_financial_query_balance_still_routes_to_engine(self, client: TestClient, ui_user_fixture: dict):
        """Test 6: Financial balance queries continue to invoke the deterministic financial engine."""
        user_id = ui_user_fixture["user_id"]
        response = client.post(
            "/ask",
            json={"user_id": user_id, "query": "What's my balance?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "structured_data" in data
        assert Decimal(str(data["structured_data"]["balance"])) == Decimal("75000.00")

    def test_scam_query_still_routes_to_scam_checker(self, client: TestClient, ui_user_fixture: dict, monkeypatch):
        """Test 7: Scam queries continue to route to the PROTECT scam checker."""
        user_id = ui_user_fixture["user_id"]

        def mock_assess(message: str, client=None, model=None):
            return {
                "risk_level": "high",
                "looks_suspicious": True,
                "indicators": [{"type": "otp_request", "evidence": "Send OTP"}],
                "explanation": "Phishing attempt.",
                "recommended_actions": ["Do not share OTP."],
                "limitations": "This is an AI pattern-based safety assessment, not a deterministic fraud verification system.",
            }

        monkeypatch.setattr("ai.pipeline.assess_scam_message", mock_assess)

        response = client.post(
            "/ask",
            json={
                "user_id": user_id,
                "query": "Is this a scam? Send your OTP immediately.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["structured_data"]["risk_level"] == "high"
        assert "⚠️" in data["answer_text"]

    def test_ui_control_intents_never_invoke_financial_engine(self, client: TestClient, ui_user_fixture: dict, monkeypatch):
        """Test 8: Invariant - UI control intents do NOT invoke any financial engine functions."""
        user_id = ui_user_fixture["user_id"]

        with patch.object(real_engine, "get_balance") as mock_bal, \
             patch.object(real_engine, "get_spending_summary") as mock_spend, \
             patch.object(real_engine, "check_affordability") as mock_afford, \
             patch.object(real_engine, "project_goal_completion") as mock_goal, \
             patch.object(real_engine, "get_insights") as mock_insight:

            for cmd, intent_name in [
                ("Sync my bank", "sync_bank"),
                ("Read my recent transactions", "read_recent_transactions"),
                ("Read my goals", "read_goals"),
                ("Upload a document", "upload_document"),
            ]:
                monkeypatch.setattr(
                    "ai.pipeline.route_query",
                    lambda *args, _in=intent_name, **kwargs: {"status": "success", "function_name": _in, "arguments": {}},
                )
                res = client.post("/ask", json={"user_id": user_id, "query": cmd})
                assert res.status_code == 200

            assert mock_bal.call_count == 0
            assert mock_spend.call_count == 0
            assert mock_afford.call_count == 0
            assert mock_goal.call_count == 0
            assert mock_insight.call_count == 0

    def test_structured_data_frontend_contract(self, client: TestClient, ui_user_fixture: dict, monkeypatch):
        """Test 9: Verify structured_data format contains action, intent, and status."""
        user_id = ui_user_fixture["user_id"]
        monkeypatch.setattr(
            "ai.pipeline.route_query",
            lambda *args, **kwargs: {"status": "success", "function_name": "sync_bank", "arguments": {}},
        )
        response = client.post(
            "/ask",
            json={"user_id": user_id, "query": "Sync my bank"},
        )
        assert response.status_code == 200
        data = response.json()
        struct = data["structured_data"]
        assert struct["action"] == "sync_bank"
        assert struct["intent"] == "sync_bank"
        assert struct["status"] == "success"

    def test_multiturn_conversation_with_ui_control_command(self, client: TestClient, ui_user_fixture: dict):
        """Test 10: Multi-turn flow where user switches or responds with a UI command."""
        user_id = ui_user_fixture["user_id"]

        # Turn 1: User asks generic question -> clarification needed
        turn1 = client.post(
            "/ask",
            json={"user_id": user_id, "query": "Can you help me?"},
        )
        assert turn1.status_code == 200
        data1 = turn1.json()
        conv_id = data1["conversation_id"]

        # Turn 2: User responds with UI command
        turn2 = client.post(
            "/ask",
            json={"user_id": user_id, "query": "Sync my bank", "conversation_id": conv_id},
        )
        assert turn2.status_code == 200
        data2 = turn2.json()
        assert data2["conversation_status"] == "completed"
        assert data2["structured_data"]["action"] == "sync_bank"
        assert data2["structured_data"]["intent"] == "sync_bank"
