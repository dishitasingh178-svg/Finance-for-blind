"""
FinSight AI Layer Package.

Exports the Intent Router, Explainer, and Pipeline components.
"""

from ai.intent_router import route_intent
from ai.explainer import explain
from ai.pipeline import AIPipeline
from ai.fake_engine import FakeFinancialEngine

__all__ = [
    "route_intent",
    "explain",
    "AIPipeline",
    "FakeFinancialEngine",
]
