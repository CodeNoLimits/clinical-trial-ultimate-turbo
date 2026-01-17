"""
AI Explainability Module

Implements FDA/EMA compliant AI explainability features:
- SHAP-based feature importance (when applicable)
- LLM narrative generation
- Explainability table generation
- Audit trail management
"""

from .explainability_table import ExplainabilityTableGenerator
from .narrative_generator import NarrativeGenerator

__all__ = [
    "ExplainabilityTableGenerator",
    "NarrativeGenerator",
]
