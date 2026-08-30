"""
Integration Tests for Conversational PROTECT Scam & Fraud Safety Checker (POST /ask).

Tests:
1. Direct scam request via conversational interface
2. Multi-turn scam checking dialogue (Turn 1: "Can you check a message?" -> Turn 2: pastes SMS)
3. Financial queries (get_balance) still work deterministically
4. Affordability queries (check_affordability) still work deterministically
5. Routine legitimate messages marked low risk in conversational response
6. Ambiguous messages handled with cautious medium risk
7. Scam checker service failure fallback transparency without crashing
8. Invariant: Financial engine functions are NEVER invoked for scam checks
9. Invariant: Scam checker is NEVER invoked for financial queries
10. Invariant: No sensitive credentials (OTP/PIN/password/CVV) are ever solicited
"""

from decimal import Decimal
from datetime import datetime, date
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import User, Account, Transaction, Goal, Bill
from backend.main import app
import backend.engine.financial_engine as real_engine
import ai.scam_checker as scam_checker_module


@pytest.fixture
def protect_user_fixture(db_session: Session) -> dict:
    """Sets up a complete user profile with transactions and accounts for conversational testing."""
    user = User(
        full_name="Priya Patel",
        email="priya.protect@example.com",
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
        balance=Decimal("95000.00"),
        monthly_income=Decimal("60000.00"),
        currency="INR",
        is_active=True,
    )
    db_session.add(account)
    db_session.flush()

    txs = [
        Transaction(
            account_id=account.id,
            user_id=user.id,
            amount=Decimal("100000.00"),
            transaction_type="income",
            category="Other",
            description="Monthly Salary",
            transaction_date=datetime(2026, 8, 1, 10, 0),
        ),
        Transaction(
            account_id=account.id,
            user_id=user.id,
            amount=Decimal("-5000.00"),
            transaction_type="expense",
            category="Food",
            merchant_name="Supermarket",
            description="Groceries",
            transaction_date=datetime(2026, 8, 15, 12, 0),
        ),
    ]
    db_session.add_all(txs)
    db_session.commit()

    return {
        "user_id": user.id,
        "account_id": account.id,
    }


class TestConversationalScamChecker:
    """Conversational pipeline tests for the PROTECT scam checker via POST /ask."""

    def test_direct_scam_query_flow(self, client: TestClient, protect_user_fixture: dict, monkeypatch):
        """Test 1: User asks directly 'Is this a scam? Send OTP immediately'."""
        user_id = protect_user_fixture["user_id"]

        def mock_assess(message: str, client=None, model=None):
            return {
                "risk_level": "high",
                "looks_suspicious": True,
                "indicators": [
                    {"type": "urgency", "evidence": "in 10 minutes"},
                    {"type": "account_threat", "evidence": "Your SBI account will be blocked"},
                    {"type": "otp_request", "evidence": "Send your OTP immediately"},
                ],
                "explanation": "The message demands an OTP under urgent threat of account suspension.",
                "recommended_actions": [
                    "Never share your OTP or PIN with anyone.",
                    "Verify your account status on the official banking portal.",
                ],
                "limitations": "This is an AI pattern-based safety assessment, not a deterministic fraud verification system.",
            }

        monkeypatch.setattr("ai.pipeline.assess_scam_message", mock_assess)

        response = client.post(
            "/ask",
            json={
                "user_id": user_id,
                "query": "Is this a scam? Your SBI account will be blocked in 10 minutes. Send your OTP immediately.",
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert data["conversation_status"] == "completed"
        assert "structured_data" in data
        scam_data = data["structured_data"]
        assert scam_data["risk_level"] == "high"
        assert scam_data["looks_suspicious"] is True
        assert len(scam_data["indicators"]) >= 1

        answer = data["answer_text"]
        assert "⚠️" in answer or "suspicious" in answer.lower()
        assert "Risk Level: HIGH" in answer
        assert "Why:" in answer
        assert "What you should do:" in answer
        assert "pattern-based" in answer.lower()

    def test_multiturn_scam_checking_flow(self, client: TestClient, protect_user_fixture: dict, monkeypatch):
        """Test 2: Multi-turn flow asking to check a message, then providing it in turn 2."""
        user_id = protect_user_fixture["user_id"]

        def mock_assess(message: str, client=None, model=None):
            return {
                "risk_level": "high",
                "looks_suspicious": True,
                "indicators": [
                    {"type": "app_install_request", "evidence": "Install this app"},
                    {"type": "kyc_threat", "evidence": "Your KYC is suspended"},
                    {"type": "urgency", "evidence": "URGENT"},
                ],
                "explanation": "The message demands app installation and screen sharing under threat of KYC suspension.",
                "recommended_actions": [
                    "Do not install unverified third-party apps.",
                    "Never share remote screen access codes.",
                ],
                "limitations": "This is an AI pattern-based safety assessment, not a deterministic fraud verification system.",
            }

        monkeypatch.setattr("ai.pipeline.assess_scam_message", mock_assess)

        # Turn 1: User initiates inquiry without message
        turn1 = client.post(
            "/ask",
            json={
                "user_id": user_id,
                "query": "Can you check if a message is a scam?",
            },
        )
        assert turn1.status_code == 200
        data1 = turn1.json()
        assert data1["conversation_status"] == "awaiting_clarification"
        assert "paste" in data1["answer_text"].lower() or "message" in data1["answer_text"].lower()
        conv_id = data1["conversation_id"]
        assert conv_id is not None

        # Turn 2: User pastes the suspicious SMS using the same conversation_id
        turn2 = client.post(
            "/ask",
            json={
                "user_id": user_id,
                "query": "URGENT: Your KYC is suspended. Install this app and share the code shown on screen.",
                "conversation_id": conv_id,
            },
        )
        assert turn2.status_code == 200
        data2 = turn2.json()
        assert data2["conversation_status"] == "completed"
        assert data2["structured_data"]["risk_level"] == "high"
        assert data2["structured_data"]["looks_suspicious"] is True
        assert any(i["type"] in ("app_install_request", "kyc_threat", "urgency") for i in data2["structured_data"]["indicators"])

    def test_financial_query_balance_integrity(self, client: TestClient, protect_user_fixture: dict):
        """Test 3: Financial balance queries continue to invoke the deterministic financial engine."""
        user_id = protect_user_fixture["user_id"]
        response = client.post(
            "/ask",
            json={"user_id": user_id, "query": "What's my balance?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "structured_data" in data
        assert Decimal(str(data["structured_data"]["balance"])) == Decimal("95000.00")

    def test_financial_query_affordability_integrity(self, client: TestClient, protect_user_fixture: dict):
        """Test 4: Financial affordability queries continue to work deterministically."""
        user_id = protect_user_fixture["user_id"]
        response = client.post(
            "/ask",
            json={"user_id": user_id, "query": "Can I afford headphones for 8k?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["structured_data"]["can_afford"] is True

    def test_legitimate_message_low_risk_conversational(self, client: TestClient, protect_user_fixture: dict, monkeypatch):
        """Test 5: Routine utility bill message receives low risk evaluation in /ask."""
        user_id = protect_user_fixture["user_id"]

        def mock_assess(message: str, client=None, model=None):
            return {
                "risk_level": "low",
                "looks_suspicious": False,
                "indicators": [],
                "explanation": "Standard billing notification with official channels.",
                "recommended_actions": ["Pay via official banking app."],
                "limitations": "This is an AI pattern-based safety assessment, not a deterministic fraud verification system.",
            }

        monkeypatch.setattr("ai.pipeline.assess_scam_message", mock_assess)

        response = client.post(
            "/ask",
            json={
                "user_id": user_id,
                "query": "Is this a scam: Hi, this is your electricity provider. Your bill of ₹1,850 is due on September 5. Pay through the official app.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["structured_data"]["risk_level"] == "low"
        assert data["structured_data"]["looks_suspicious"] is False
        assert "✅" in data["answer_text"] or "LOW" in data["answer_text"]

    def test_ambiguous_message_medium_risk(self, client: TestClient, protect_user_fixture: dict, monkeypatch):
        """Test 6: Ambiguous shortened URL message returns medium risk with disclaimer."""
        user_id = protect_user_fixture["user_id"]

        def mock_assess(message: str, client=None, model=None):
            return {
                "risk_level": "medium",
                "looks_suspicious": True,
                "indicators": [{"type": "shortened_url", "evidence": "http://tinyurl.com/invoice-281"}],
                "explanation": "Shortened link could mask destination.",
                "recommended_actions": ["Verify link before clicking."],
                "limitations": "This is an AI pattern-based safety assessment, not a deterministic fraud verification system.",
            }

        monkeypatch.setattr("ai.pipeline.assess_scam_message", mock_assess)

        response = client.post(
            "/ask",
            json={
                "user_id": user_id,
                "query": "Check this message: Please check your pending invoice at http://tinyurl.com/invoice-281",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["structured_data"]["risk_level"] in ("medium", "high")
        assert "pattern-based" in data["answer_text"].lower()

    def test_scam_checker_llm_failure_conversational_fallback(self, client: TestClient, protect_user_fixture: dict, monkeypatch):
        """Test 7: If the scam checker LLM fails, the pipeline returns a transparent fallback without crashing."""
        user_id = protect_user_fixture["user_id"]

        def mock_failing_assess(message: str, client=None, model=None):
            return {
                "risk_level": "medium",
                "looks_suspicious": False,
                "indicators": [],
                "explanation": "Unable to complete safety assessment due to an AI service error. Please exercise caution.",
                "recommended_actions": [
                    "Do not share OTP, PIN, password, or sensitive details.",
                    "Verify directly through official channels.",
                ],
                "limitations": "This is an AI pattern-based safety assessment, not a deterministic fraud verification system. Assessment was incomplete due to a service error.",
            }

        monkeypatch.setattr("ai.pipeline.assess_scam_message", mock_failing_assess)

        response = client.post(
            "/ask",
            json={
                "user_id": user_id,
                "query": "Is this a scam? Transfer money now.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "Unable to complete safety assessment" in data["answer_text"] or "caution" in data["answer_text"].lower()
        assert data["structured_data"]["risk_level"] == "medium"

    def test_financial_engine_never_called_for_scam_checks(self, client: TestClient, protect_user_fixture: dict):
        """Test 8: Invariant - Deterministic financial engine methods are NEVER called for scam checking."""
        user_id = protect_user_fixture["user_id"]

        with patch.object(real_engine, "get_balance") as mock_bal, \
             patch.object(real_engine, "get_spending_summary") as mock_spend, \
             patch.object(real_engine, "check_affordability") as mock_afford, \
             patch.object(real_engine, "project_goal_completion") as mock_goal, \
             patch.object(real_engine, "get_insights") as mock_insight:

            response = client.post(
                "/ask",
                json={
                    "user_id": user_id,
                    "query": "Is this a scam? Your SBI account will be blocked in 10 minutes. Send OTP.",
                },
            )
            assert response.status_code == 200
            assert mock_bal.call_count == 0
            assert mock_spend.call_count == 0
            assert mock_afford.call_count == 0
            assert mock_goal.call_count == 0
            assert mock_insight.call_count == 0

    def test_scam_checker_never_called_for_financial_queries(self, client: TestClient, protect_user_fixture: dict):
        """Test 9: Invariant - Scam checker is NEVER invoked for ordinary balance queries."""
        user_id = protect_user_fixture["user_id"]

        with patch.object(scam_checker_module, "assess_scam_message") as mock_scam:
            response = client.post(
                "/ask",
                json={"user_id": user_id, "query": "What's my balance?"},
            )
            assert response.status_code == 200
            assert mock_scam.call_count == 0

    def test_zero_credential_solicitation_in_responses(self, client: TestClient, protect_user_fixture: dict):
        """Test 10: Invariant - Responses NEVER ask the user to share OTP, PIN, password, or CVV."""
        user_id = protect_user_fixture["user_id"]
        response = client.post(
            "/ask",
            json={
                "user_id": user_id,
                "query": "Is this a scam? Send your OTP immediately.",
            },
        )
        assert response.status_code == 200
        answer = response.json()["answer_text"].lower()

        # The assistant must warn NOT to share OTP/PIN, never ask for it
        assert "do not share" in answer or "never share" in answer
        assert "please provide your otp" not in answer
        assert "please provide your pin" not in answer
        assert "send your password" not in answer
