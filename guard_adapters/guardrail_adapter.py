"""
Guardrail Adapter

This module provides a unified interface for accessing different guardrails.
"""
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass


@dataclass
class GuardResult:
    """Standardized result from guardrail evaluation."""
    provider: str
    allowed: bool
    confidence: Optional[float]


class GuardrailAdapter(ABC):
    """
    Guardrail Adapter interface.
    """
    @abstractmethod
    def check(self, text: str) -> GuardResult:
        pass
