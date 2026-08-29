"""
LLM Client for FinSight AI Layer.

Integrates with Gemini via standard OpenAI-compatible endpoints or Google AI Studio.
Configured via environment variables:
- LLM_API_KEY (or GEMINI_API_KEY)
- LLM_BASE_URL (defaults to Google Generative Language OpenAI-compatible endpoint)
- LLM_MODEL (defaults to gemini-2.5-flash)

ARCHITECTURAL PRINCIPLES:
- Zero financial calculation in LLM.
- LLM is used strictly for Natural Language Intent Routing (Function/Tool Calling)
  and Grounded Narration of backend facts.
- Graceful degradation: If LLM is unreachable or unconfigured, reports error and
  signals pipeline to switch to MOCK_FALLBACK.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
import httpx

logger = logging.getLogger(__name__)

# Tools definition for OpenAI-compatible Gemini tool calling
FINSIGHT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_balance",
            "description": "Fetch the user's current authoritative balance and timestamp.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_spending_summary",
            "description": "Fetch spending summary across all categories with month-over-month trend.",
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "enum": ["this_month", "last_month"],
                        "description": "Calendar period for spending analysis.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional specific category (Food, Transport, Shopping, Bills, Entertainment, Healthcare, Education, Other).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_affordability",
            "description": "Check if a proposed purchase amount can be afforded considering current balance and upcoming bills.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "string",
                        "description": "Monetary cost of the item in INR (e.g. '8000', '5000.50').",
                    },
                    "item_name": {
                        "type": "string",
                        "description": "Name or description of the item being purchased (e.g. 'headphones', 'bicycle').",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_goal_completion",
            "description": "Project completion timeline (months remaining) for a savings goal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_name": {
                        "type": "string",
                        "description": "Name of the savings goal (e.g. 'Emergency Fund', 'Vacation').",
                    },
                    "hypothetical_contribution": {
                        "type": "string",
                        "description": "Optional simulated monthly contribution amount.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_insights",
            "description": "Fetch financial anomalies, spending spikes, subscription increases, and upcoming bill alerts.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "payment_preview",
            "description": "Preview a payment to a recipient, check affordability and risk, and stage confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "string",
                        "description": "Amount to transfer in INR (e.g. '5000').",
                    },
                    "recipient_name": {
                        "type": "string",
                        "description": "Name of the payee or recipient (e.g. 'Dr Rao', 'Rahul').",
                    },
                },
                "required": ["amount", "recipient_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "payment_execute",
            "description": "Execute a previously staged pending payment after explicit user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pending_payment_id": {
                        "type": "string",
                        "description": "The ID or token of the pending payment to confirm.",
                    },
                    "confirmation_token": {
                        "type": "string",
                        "description": "Optional confirmation token.",
                    },
                },
                "required": [],
            },
        },
    },
]


class LLMClient:
    """
    Client for interacting with Gemini / OpenAI-compatible LLM endpoints.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 10.0,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        if not self.base_url.endswith("/"):
            self.base_url += "/"
        self.model = model or os.getenv("LLM_MODEL", "gemini-2.5-flash")
        self.timeout_seconds = timeout_seconds

    def is_available(self) -> bool:
        """Returns True if an API key is configured."""
        return bool(self.api_key and self.api_key.strip())

    def call_tool_router(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Uses Gemini Function Calling to map natural language query to a structured intent.

        Returns:
            Tuple: (intent_data_dict, error_string)
            If successful: ({"intent": ..., "arguments": {...}}, None)
            If failed: (None, error_reason)
        """
        if not self.is_available():
            return None, "LLM_API_KEY is not configured."

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the FinSight Financial AI Intent Router. "
                    "Classify user inquiries and commands into exact financial tool calls. "
                    "Do NOT calculate balances, affordability, or risks. "
                    "Extract entities: amount, category, recipient_name, goal_name, pending_payment_id. "
                    "If the user confirms an existing payment, call payment_execute."
                ),
            }
        ]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": query})

        endpoint = f"{self.base_url}chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": FINSIGHT_TOOLS,
            "tool_choice": "auto",
            "temperature": 0.1,
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(endpoint, json=payload, headers=headers)

            if response.status_code != 200:
                err_msg = f"LLM API returned status {response.status_code}: {response.text[:200]}"
                logger.warning(err_msg)
                return None, err_msg

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return None, "No choices returned from LLM."

            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls", [])

            if tool_calls:
                t_call = tool_calls[0]
                fn_name = t_call.get("function", {}).get("name", "unknown")
                raw_args = t_call.get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    args = {}

                return {
                    "intent": fn_name,
                    "arguments": args,
                }, None

            # If no tool was selected, return unknown intent with content
            content = message.get("content", "")
            return {
                "intent": "unknown",
                "arguments": {"raw_query": query, "model_text": content},
            }, None

        except httpx.TimeoutException:
            err = f"LLM request timed out after {self.timeout_seconds}s."
            logger.warning(err)
            return None, err
        except Exception as e:
            err = f"LLM routing failed with exception: {type(e).__name__}: {str(e)}"
            logger.warning(err)
            return None, err

    def explain_facts(
        self,
        intent: str,
        facts: Dict[str, Any],
        query: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Uses Gemini to generate an accessible, screen-reader-friendly narration
        grounded strictly in the authoritative facts.

        Returns:
            Tuple: (answer_text, aria_priority, error_string)
        """
        if not self.is_available():
            return None, None, "LLM_API_KEY is not configured."

        system_prompt = (
            "You are FinSight Accessible AI Explainer. "
            "Your job is to clearly and concisely narrate authoritative financial facts for a screen reader / voice user. "
            "STRICT RULES:\n"
            "1. You MUST ONLY state the numbers and values provided in the JSON facts. NEVER compute, adjust, or guess new numbers.\n"
            "2. Always format currency in INR with ₹ (e.g. ₹5,000.00).\n"
            "3. If facts indicate requires_confirmation is True, explicitly tell the user to confirm.\n"
            "4. If facts indicate a fraud_warning or high risk, highlight it clearly.\n"
            "5. If facts indicate clarification_needed, politely ask the clarification question.\n"
            "6. Output ONLY your narration text."
        )

        user_content = f"User Query: {query or ''}\nIntent: {intent}\nAuthoritative Facts:\n{json.dumps(facts, default=str)}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        endpoint = f"{self.base_url}chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(endpoint, json=payload, headers=headers)

            if response.status_code != 200:
                err_msg = f"LLM Explainer returned status {response.status_code}: {response.text[:200]}"
                return None, None, err_msg

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return None, None, "No choices returned from LLM."

            text = choices[0].get("message", {}).get("content", "").strip()

            # Determine ARIA priority
            aria_priority = "polite"
            if facts.get("fraud_warning") or facts.get("can_afford") is False or facts.get("risk_level") == "high":
                aria_priority = "assertive"

            return text, aria_priority, None

        except Exception as e:
            err = f"LLM Explainer failed: {str(e)}"
            return None, None, err
