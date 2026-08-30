"""
FinSight AI Scam & Fraud Safety Checker Unit & Integration Tests (PROTECT Pillar).

Tests:
1. OTP scam detection
2. Urgent bank-blocking scam detection
3. Fake prize / refund scam detection
4. Suspicious payment request detection
5. Normal legitimate-looking message assessment
6. Ambiguous message handling
7. Empty / whitespace message handling
8. LLM API / service failure fallback
9. Malformed / invalid model output fallback
10. Explicit limitations disclaimer verification across all responses
11. FastAPI endpoint POST /protect/scam-check
12. FastAPI versioned endpoint POST /api/v1/protect/scam-check
13. Safety invariant: no solicitation of sensitive credentials in recommendations
"""

import json
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from ai.scam_checker import assess_scam_message, SCAM_CHECKER_SYSTEM_PROMPT
from backend.main import app


def _create_mock_llm_client(return_content: str) -> MagicMock:
    """Helper to create a mocked OpenAI-compatible client returning specified content."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = return_content
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def _create_failing_llm_client(exception: Exception) -> MagicMock:
    """Helper to create a mocked OpenAI client that raises an exception."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = exception
    return mock_client


class TestScamSafetyCheckerUnit:
    """Unit tests for assess_scam_message."""

    def test_otp_scam_detection(self):
        """Test 1: Identifies high risk OTP phishing message."""
        mock_output = json.dumps({
            "risk_level": "high",
            "looks_suspicious": True,
            "indicators": [
                {"type": "otp_request", "evidence": "Send your OTP immediately to verify"},
                {"type": "urgency", "evidence": "blocked in 10 minutes"},
                {"type": "account_threat", "evidence": "Your SBI account will be blocked"},
            ],
            "explanation": "The message creates artificial urgency and explicitly demands an OTP under threat of account blocking.",
            "recommended_actions": [
                "Never share your OTP or PIN with anyone.",
                "Verify your account status by logging into the official SBI mobile app.",
            ],
            "limitations": "This is an AI pattern-based safety assessment, not a deterministic fraud verification system.",
        })
        mock_client = _create_mock_llm_client(mock_output)

        result = assess_scam_message(
            message="Your SBI account will be blocked in 10 minutes. Send your OTP immediately to verify.",
            client=mock_client,
        )

        assert result["risk_level"] == "high"
        assert result["looks_suspicious"] is True
        assert len(result["indicators"]) >= 2
        indicator_types = [i["type"] for i in result["indicators"]]
        assert "otp_request" in indicator_types
        assert "urgency" in indicator_types
        assert "OTP" in result["recommended_actions"][0] or "PIN" in result["recommended_actions"][0]
        assert "pattern-based" in result["limitations"].lower()

    def test_urgent_bank_blocking_scam(self):
        """Test 2: Identifies bank suspension and remote-access app installation threats."""
        mock_output = json.dumps({
            "risk_level": "high",
            "looks_suspicious": True,
            "indicators": [
                {"type": "kyc_threat", "evidence": "Your KYC is suspended"},
                {"type": "app_install_request", "evidence": "Install this app and share the code shown on screen"},
                {"type": "urgency", "evidence": "URGENT"},
            ],
            "explanation": "The message uses urgent fear tactics claiming KYC suspension and instructs installing an unknown application.",
            "recommended_actions": [
                "Do not install apps or APKs from unknown links or messages.",
                "Do not share any screen sharing or remote codes.",
                "Visit your official bank branch or app for KYC updates.",
            ],
            "limitations": "This is an AI pattern-based safety assessment, not a deterministic fraud verification system.",
        })
        mock_client = _create_mock_llm_client(mock_output)

        result = assess_scam_message(
            message="URGENT: Your KYC is suspended. Install this app and share the code shown on screen.",
            client=mock_client,
        )

        assert result["risk_level"] == "high"
        assert result["looks_suspicious"] is True
        assert any(i["type"] in ("app_install_request", "kyc_threat") for i in result["indicators"])
        assert "install" in result["explanation"].lower() or "kyc" in result["explanation"].lower()

    def test_fake_prize_refund_scam(self):
        """Test 3: Identifies prize/lottery scam requiring upfront advance fee."""
        mock_output = json.dumps({
            "risk_level": "high",
            "looks_suspicious": True,
            "indicators": [
                {"type": "fake_reward", "evidence": "Congratulations! You won ₹25,000"},
                {"type": "payment_demand", "evidence": "Pay ₹499 processing fee to claim your prize"},
            ],
            "explanation": "The message promises a cash prize but demands an upfront processing fee, which is a classic advance-fee scam pattern.",
            "recommended_actions": [
                "Do not pay any processing fee or advance amount to claim prizes.",
                "Block and report the sender.",
            ],
            "limitations": "This is an AI pattern-based safety assessment, not a deterministic fraud verification system.",
        })
        mock_client = _create_mock_llm_client(mock_output)

        result = assess_scam_message(
            message="Congratulations! You won ₹25,000. Pay ₹499 processing fee to claim your prize.",
            client=mock_client,
        )

        assert result["risk_level"] == "high"
        assert result["looks_suspicious"] is True
        assert any(i["type"] in ("fake_reward", "payment_demand") for i in result["indicators"])
        assert "499" in result["indicators"][1]["evidence"] or "fee" in result["indicators"][1]["evidence"]

    def test_suspicious_payment_request(self):
        """Test 4: Identifies urgent suspicious money transfer requests."""
        mock_output = json.dumps({
            "risk_level": "high",
            "looks_suspicious": True,
            "indicators": [
                {"type": "payment_demand", "evidence": "transfer ₹5,000 to this UPI ID immediately"},
                {"type": "urgency", "evidence": "emergency medical need"},
            ],
            "explanation": "The sender claims an emergency and demands an immediate UPI transfer without prior verification.",
            "recommended_actions": [
                "Call the person directly on their known phone number to verify their identity.",
                "Do not transfer funds to unfamiliar UPI handles.",
            ],
            "limitations": "This is an AI pattern-based safety assessment, not a deterministic fraud verification system.",
        })
        mock_client = _create_mock_llm_client(mock_output)

        result = assess_scam_message(
            message="Hey, I'm in an emergency medical need, please transfer ₹5,000 to this UPI ID immediately: emergency@upi",
            client=mock_client,
        )

        assert result["risk_level"] in ("high", "medium")
        assert result["looks_suspicious"] is True
        assert len(result["indicators"]) >= 1

    def test_normal_legitimate_informational_message(self):
        """Test 5: Correctly marks a routine utility bill notification as low risk."""
        mock_output = json.dumps({
            "risk_level": "low",
            "looks_suspicious": False,
            "indicators": [],
            "explanation": "The message provides standard billing information with a future due date and directs payment via the official app.",
            "recommended_actions": [
                "Pay through the provider's verified official mobile app or official website.",
            ],
            "limitations": "This is an AI pattern-based safety assessment, not a deterministic fraud verification system.",
        })
        mock_client = _create_mock_llm_client(mock_output)

        result = assess_scam_message(
            message="Hi, this is your electricity provider. Your bill of ₹1,850 is due on September 5. Pay through the official app.",
            client=mock_client,
        )

        assert result["risk_level"] == "low"
        assert result["looks_suspicious"] is False
        assert len(result["indicators"]) == 0
        assert "official app" in result["explanation"].lower() or "standard" in result["explanation"].lower() or "low" in result["risk_level"]

    def test_ambiguous_message(self):
        """Test 6: Handles ambiguous message with cautious medium risk and limited confidence."""
        mock_output = json.dumps({
            "risk_level": "medium",
            "looks_suspicious": True,
            "indicators": [
                {"type": "suspicious_link", "evidence": "http://tinyurl.com/order-info-982"}
            ],
            "explanation": "The message includes a shortened URL with limited context. Confidence is limited as sender identity cannot be verified.",
            "recommended_actions": [
                "Do not click on shortened or unverified URLs.",
                "Check your order status directly on the merchant's official portal.",
            ],
            "limitations": "This is an AI pattern-based safety assessment, not a deterministic fraud verification system.",
        })
        mock_client = _create_mock_llm_client(mock_output)

        result = assess_scam_message(
            message="Please check your pending invoice here: http://tinyurl.com/order-info-982",
            client=mock_client,
        )

        assert result["risk_level"] in ("medium", "high")
        assert "suspicious_link" in [i["type"] for i in result["indicators"]]
        assert "pattern-based" in result["limitations"].lower()

    def test_empty_message_handling(self):
        """Test 7: Handles empty and whitespace-only messages without calling LLM or crashing."""
        result_empty = assess_scam_message(message="")
        assert result_empty["risk_level"] == "low"
        assert result_empty["looks_suspicious"] is False
        assert "No message text" in result_empty["explanation"]

        result_whitespace = assess_scam_message(message="   \n\t  ")
        assert result_whitespace["risk_level"] == "low"
        assert result_whitespace["looks_suspicious"] is False

    def test_llm_failure_transparent_fallback(self):
        """Test 8: Handles LLM client exception with transparent fallback without pretending evaluation succeeded."""
        failing_client = _create_failing_llm_client(RuntimeError("Connection to AI gateway timed out (503)"))

        result = assess_scam_message(
            message="Your account is blocked. Send OTP.",
            client=failing_client,
        )

        assert result["risk_level"] == "medium"
        assert result["looks_suspicious"] is False
        assert "Unable to complete safety assessment" in result["explanation"]
        assert "AI service error" in result["explanation"]
        assert "pattern-based" in result["limitations"].lower()
        # Verify recommended actions still offer protective safety advice
        assert len(result["recommended_actions"]) > 0

    def test_malformed_model_output(self):
        """Test 9: Handles non-JSON and malformed responses from LLM safely."""
        malformed_client = _create_mock_llm_client("I think this message is definitely a scam! It asks for money.")

        result = assess_scam_message(
            message="Send money now to get a loan.",
            client=malformed_client,
        )

        assert "Unable to complete safety assessment" in result["explanation"]
        assert "pattern-based" in result["limitations"].lower()
        assert len(result["recommended_actions"]) > 0

    def test_markdown_wrapped_json_parsing(self):
        """Test 9b: Successfully parses JSON wrapped inside markdown code blocks."""
        markdown_json = """```json
{
  "risk_level": "high",
  "looks_suspicious": true,
  "indicators": [
    {"type": "urgency", "evidence": "within 5 mins"}
  ],
  "explanation": "Urgent timeline detected.",
  "recommended_actions": ["Do not hurry."],
  "limitations": "This is an AI pattern-based safety assessment, not a deterministic fraud verification system."
}
```"""
        mock_client = _create_mock_llm_client(markdown_json)

        result = assess_scam_message(
            message="Pay within 5 mins or service will be cut.",
            client=mock_client,
        )

        assert result["risk_level"] == "high"
        assert result["looks_suspicious"] is True
        assert len(result["indicators"]) == 1

    def test_limitations_explicit_disclaimer_invariant(self):
        """Test 10: Verifies every code path returns the mandatory limitations disclaimer."""
        mock_output = json.dumps({
            "risk_level": "low",
            "looks_suspicious": False,
            "indicators": [],
            "explanation": "Routine message.",
            "recommended_actions": ["No action required."],
            # Deliberately omit limitations in raw output to test automatic enforcement
        })
        mock_client = _create_mock_llm_client(mock_output)

        result = assess_scam_message(
            message="Meeting at 4pm today.",
            client=mock_client,
        )

        assert "limitations" in result
        assert "pattern-based" in result["limitations"].lower()
        assert "not a deterministic fraud" in result["limitations"].lower()

    def test_no_sensitive_credentials_solicited_in_prompt_and_code(self):
        """Test 13: Invariant test verifying system prompt strictly forbids asking for credentials."""
        prompt_lower = SCAM_CHECKER_SYSTEM_PROMPT.lower()
        assert "never ask the user" in prompt_lower or "no credential solicitation" in prompt_lower
        assert "otp" in prompt_lower
        assert "pin" in prompt_lower
        assert "cvv" in prompt_lower
        assert "password" in prompt_lower


class TestScamSafetyCheckerAPI:
    """Integration tests for FastAPI endpoints POST /protect/scam-check and /api/v1/protect/scam-check."""

    @pytest.fixture
    def test_client(self) -> TestClient:
        return TestClient(app)

    def test_api_post_protect_scam_check_success(self, test_client: TestClient, monkeypatch):
        """Test 11: POST /protect/scam-check returns 200 with structured JSON contract."""
        def mock_assess(message: str, client=None, model=None):
            return {
                "risk_level": "high",
                "looks_suspicious": True,
                "indicators": [
                    {"type": "account_threat", "evidence": "Your account will be blocked today"},
                    {"type": "otp_request", "evidence": "Send OTP immediately"},
                ],
                "explanation": "The message demands an immediate OTP under threat of account blocking.",
                "recommended_actions": [
                    "Never share your OTP with anyone.",
                    "Verify with your bank via their official app.",
                ],
                "limitations": "This is an AI pattern-based safety assessment, not a deterministic fraud verification system.",
            }

        monkeypatch.setattr("backend.routers.protect.assess_scam_message", mock_assess)

        response = test_client.post(
            "/protect/scam-check",
            json={
                "user_id": 1,
                "message": "Your account will be blocked today. Send OTP immediately.",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["risk_level"] == "high"
        assert data["looks_suspicious"] is True
        assert len(data["indicators"]) == 2
        assert data["indicators"][0]["type"] == "account_threat"
        assert "pattern-based" in data["limitations"].lower()

    def test_api_versioned_endpoint_parity(self, test_client: TestClient, monkeypatch):
        """Test 12: POST /api/v1/protect/scam-check parity."""
        def mock_assess(message: str, client=None, model=None):
            return {
                "risk_level": "low",
                "looks_suspicious": False,
                "indicators": [],
                "explanation": "Legitimate electricity bill notice.",
                "recommended_actions": ["Pay via official app."],
                "limitations": "This is an AI pattern-based safety assessment, not a deterministic fraud verification system.",
            }

        monkeypatch.setattr("backend.routers.protect.assess_scam_message", mock_assess)

        response = test_client.post(
            "/api/v1/protect/scam-check",
            json={
                "message": "Hi, this is your electricity provider. Your bill of ₹1,850 is due on September 5.",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["risk_level"] == "low"
        assert data["looks_suspicious"] is False

    def test_api_empty_message_validation_422(self, test_client: TestClient):
        """Validates that empty message payloads are rejected with HTTP 422 by Pydantic."""
        response = test_client.post(
            "/protect/scam-check",
            json={"message": ""},
        )
        assert response.status_code == 422
