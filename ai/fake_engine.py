"""
Fake Engine for FinSight AI Layer Isolation Testing.

Provides deterministic static responses mimicking the backend engine
for isolated unit testing of the AI Intent Router and Explainer.
"""

from decimal import Decimal
from datetime import datetime
from typing import Dict, Any, Optional


class FakeFinancialEngine:
    """Mock financial and payment engine for standalone AI tests."""

    @staticmethod
    def get_balance(user_id: int = 1) -> Dict[str, Any]:
        return {
            "intent": "get_balance",
            "balance": Decimal("138372.00"),
            "as_of": datetime(2026, 8, 27, 12, 0, 0),
        }

    @staticmethod
    def get_spending_summary(user_id: int = 1, period: str = "this_month", category: Optional[str] = None) -> Dict[str, Any]:
        by_cat = {
            "Food": Decimal("27330.00"),
            "Transport": Decimal("4500.00"),
            "Shopping": Decimal("12000.00"),
            "Bills": Decimal("8500.00"),
            "Entertainment": Decimal("3200.00"),
            "Healthcare": Decimal("1500.00"),
            "Education": Decimal("0.00"),
            "Other": Decimal("11400.00"),
        }
        vs_last = {cat: Decimal("17.50") if cat == "Food" else Decimal("0.00") for cat in by_cat}
        res = {
            "intent": "get_spending_summary",
            "period": period,
            "total": sum(by_cat.values(), Decimal("0.00")),
            "by_category": by_cat,
            "vs_last_period_pct": vs_last,
        }
        if category:
            res["requested_category"] = category
        return res

    @staticmethod
    def check_affordability(user_id: int = 1, amount: Decimal = Decimal("8000.00"), item_name: Optional[str] = None) -> Dict[str, Any]:
        return {
            "intent": "check_affordability",
            "amount": amount,
            "can_afford": True,
            "balance_after": Decimal("130372.00"),
            "upcoming_bills": Decimal("2849.00"),
            "savings_goal_impact_months": Decimal("0"),
            "item_name": item_name,
            "reasoning_facts": [
                {"fact": "current_balance", "value": "138372.00"},
                {"fact": "purchase_amount", "value": str(amount)},
                {"fact": "upcoming_bills", "value": "2849.00"},
            ],
        }

    @staticmethod
    def project_goal_completion(goal_name: str = "Emergency Fund") -> Dict[str, Any]:
        return {
            "intent": "project_goal_completion",
            "goal_id": 1,
            "goal_name": goal_name,
            "target_amount": Decimal("150000.00"),
            "current_amount": Decimal("45000.00"),
            "monthly_contribution": Decimal("15000.00"),
            "current_months_remaining": Decimal("7"),
            "hypothetical_months_remaining": None,
        }

    @staticmethod
    def get_insights(user_id: int = 1) -> Dict[str, Any]:
        return {
            "intent": "get_insights",
            "insights": [
                {
                    "type": "spending_increase",
                    "category": "Food",
                    "pct": Decimal("17.50"),
                    "period": "this_month",
                },
                {
                    "type": "subscription_increase",
                    "category": "Entertainment",
                    "merchant": "Netflix",
                    "pct": Decimal("40.08"),
                    "period": "August 2026",
                },
            ],
        }
