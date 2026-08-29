"""
Tests for AI Intent Router, Explainer, Pipeline, and Multi-turn Clarifications.
"""

import json
from unittest.mock import patch, MagicMock
from decimal import Decimal

from ai.intent_router import route_intent
from ai.explainer import explain
from ai.fake_engine import FakeFinancialEngine
from ai.llm_client import LLMClient
from ai.conversation import conversation_manager, parse_shorthand_amount


class TestAIIntentRouter:
    def test_route_balance(self):
        res = route_intent("How much money do I have in my account?")
        assert res["intent"] == "get_balance"
        assert res["execution_mode"] == "MOCK_FALLBACK"

    def test_route_spending_food(self):
        res = route_intent("How much did I spend on food this month?")
        assert res["intent"] == "get_spending_summary"
        assert res["arguments"]["period"] == "this_month"
        assert res["arguments"]["category"] == "Food"

    def test_route_spending_last_month(self):
        res = route_intent("What did I spend last month?")
        assert res["intent"] == "get_spending_summary"
        assert res["arguments"]["period"] == "last_month"

    def test_route_affordability(self):
        res = route_intent("Can I afford headphones for ₹8,000?")
        assert res["intent"] == "check_affordability"
        assert res["arguments"]["amount"] == "8000"

    def test_route_affordability_shorthand_8k(self):
        res = route_intent("Can I buy a laptop for 80k?")
        assert res["intent"] == "check_affordability"
        assert res["arguments"]["amount"] == "80000"

    def test_route_affordability_missing_amount(self):
        res = route_intent("Can I afford it?")
        assert res["intent"] == "check_affordability"
        assert res["arguments"]["amount"] is None

    def test_route_goal(self):
        res = route_intent("When will I finish my Emergency Fund?")
        assert res["intent"] == "project_goal_completion"
        assert "Emergency Fund" in res["arguments"]["goal_name"]

    def test_route_insights(self):
        res = route_intent("Show me any financial insights or updates")
        assert res["intent"] == "get_insights"

    def test_route_payment_preview(self):
        res = route_intent("Send ₹5,000 to Dr Rao")
        assert res["intent"] == "payment_preview"
        assert res["arguments"]["amount"] == "5000"
        assert res["arguments"]["recipient_name"] == "Dr Rao"

    def test_route_payment_execute_confirmation(self):
        res = route_intent("Confirm payment", confirmation_token="123")
        assert res["intent"] == "payment_execute"
        assert res["arguments"]["pending_payment_id"] == "123"


class TestAIExplainer:
    def test_explain_balance(self):
        facts = FakeFinancialEngine.get_balance()
        res = explain({"intent": "get_balance"}, facts)
        assert res["intent"] == "get_balance"
        assert "₹138,372.00" in res["answer_text"]
        assert res["aria_priority"] == "polite"
        assert res["structured_facts"] == res["structured_data"]

    def test_explain_spending_category(self):
        facts = FakeFinancialEngine.get_spending_summary(category="Food")
        res = explain({"intent": "get_spending_summary"}, facts)
        assert "₹27,330.00 on Food" in res["answer_text"]
        assert "17.50%" in res["answer_text"]

    def test_explain_affordability(self):
        facts = FakeFinancialEngine.check_affordability(amount=Decimal("8000.00"), item_name="headphones")
        res = explain({"intent": "check_affordability"}, facts)
        assert "Yes, you can afford" in res["answer_text"]
        assert "₹8,000.00" in res["answer_text"]

    def test_explain_payment_preview(self):
        facts = {
            "intent": "payment_preview",
            "amount": Decimal("5000.00"),
            "recipient_name": "Dr Rao",
            "balance_after": Decimal("133372.00"),
            "upcoming_bills": Decimal("2849.00"),
            "fraud_warning": False,
            "pending_payment_id": 42,
            "requires_confirmation": True,
        }
        res = explain({"intent": "payment_preview"}, facts)
        assert res["requires_confirmation"] is True
        assert res["pending_payment_id"] == 42
        assert "₹5,000.00 to Dr Rao" in res["answer_text"]
        assert "confirm" in res["answer_text"]


class TestAILLMClientAndModes:
    def test_mock_fallback_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        client = LLMClient()
        assert not client.is_available()

        res = route_intent("How much money do I have?", llm_client=client)
        assert res["execution_mode"] == "MOCK_FALLBACK"
        assert res["intent"] == "get_balance"

    def test_real_llm_mode_with_mocked_gemini_response(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-mock-gemini-key")
        client = LLMClient()
        assert client.is_available()

        # Mock Gemini tool calling response
        mock_gemini_tool_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "check_affordability",
                                    "arguments": json.dumps({"amount": "8000", "item_name": "headphones"}),
                                }
                            }
                        ],
                    }
                }
            ]
        }

        # Mock Gemini explainer narration response
        mock_gemini_explain_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Yes, you can afford headphones for ₹8,000.00.",
                    }
                }
            ]
        }

        with patch("ai.llm_client.httpx.Client.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.side_effect = [mock_gemini_tool_resp, mock_gemini_explain_resp]
            mock_post.return_value = mock_resp

            intent_data = route_intent("Can I buy headphones for 8k?", llm_client=client)
            assert intent_data["intent"] == "check_affordability"
            assert intent_data["arguments"]["amount"] == "8000"
            assert intent_data["execution_mode"] == "REAL_LLM"

            facts = FakeFinancialEngine.check_affordability(amount=Decimal("8000.00"), item_name="headphones")
            explained = explain(intent_data=intent_data, facts=facts, query="Can I buy headphones for 8k?", llm_client=client)
            assert explained["execution_mode"] == "REAL_LLM"
            assert "Yes, you can afford" in explained["answer_text"]

    def test_mock_fallback_on_llm_server_error(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-mock-gemini-key")
        client = LLMClient()

        with patch("ai.llm_client.httpx.Client.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_resp.text = "Internal Server Error"
            mock_post.return_value = mock_resp

            intent_data = route_intent("How much did I spend on food?", llm_client=client)
            assert intent_data["intent"] == "get_spending_summary"
            assert intent_data["execution_mode"] == "MOCK_FALLBACK"
            assert "status 500" in intent_data["provider_error"]


class TestAIMultiTurnClarification:
    def setup_method(self):
        conversation_manager.clear()

    def test_clarification_affordability_8k(self):
        conv_id = "test-clarify-afford"

        # Turn 1: "Can I afford it?"
        conversation_manager.record_turn(
            conversation_id=conv_id,
            query="Can I afford it?",
            intent="check_affordability",
            arguments={"amount": None, "item_name": None},
            facts={"status": "clarification_needed", "question": "How much does the item cost?"},
            answer_text="How much does the item cost?",
        )

        # Turn 2: User answers "8k"
        resolved = route_intent("8k", conversation_id=conv_id)
        assert resolved["intent"] == "check_affordability"
        assert resolved["arguments"]["amount"] == "8000"

    def test_clarification_affordability_currency_shorthand(self):
        conv_id = "test-clarify-shorthand"
        conversation_manager.record_turn(
            conversation_id=conv_id,
            query="Can I buy the bike?",
            intent="check_affordability",
            arguments={"amount": None, "item_name": "bike"},
            facts={"status": "clarification_needed", "question": "How much does the item cost?"},
            answer_text="How much does the item cost?",
        )

        resolved = route_intent("₹12,500", conversation_id=conv_id)
        assert resolved["intent"] == "check_affordability"
        assert resolved["arguments"]["amount"] == "12500"
        assert resolved["arguments"]["item_name"] == "bike"

    def test_clarification_goal_name(self):
        conv_id = "test-clarify-goal"
        conversation_manager.record_turn(
            conversation_id=conv_id,
            query="When will I finish my goal?",
            intent="project_goal_completion",
            arguments={"goal_name": None},
            facts={"status": "clarification_needed", "question": "Which savings goal would you like me to check?"},
            answer_text="Which savings goal would you like me to check?",
        )

        resolved = route_intent("Emergency Fund", conversation_id=conv_id)
        assert resolved["intent"] == "project_goal_completion"
        assert resolved["arguments"]["goal_name"] == "Emergency Fund"


class TestShorthandParser:
    def test_parse_k(self):
        assert parse_shorthand_amount("8k") == "8000"
        assert parse_shorthand_amount("8.5k") == "8500"
        assert parse_shorthand_amount("₹10k") == "10000"
        assert parse_shorthand_amount("500") == "500"
        assert parse_shorthand_amount("₹5,000.50") == "5000.50"
