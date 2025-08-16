import math
import sys
from typing import List, Tuple

import numpy as np
from textattack.models.wrappers import ModelWrapper

from guard_adapters.guardrail_adapter import GuardrailAdapter, GuardResult


class GuardrailModelWrapper(ModelWrapper):
    """
    Wrapper to make guardrail compatible with TextAttack.

    This wrapper adapts guardrail APIs to work with TextAttack's model interface.
    """

    def __init__(self, guardrail_adapter: GuardrailAdapter):
        """
        Initialize the wrapper.

        Args:
            guardrail_adapter: Guardrail adapter instance
        """

        self.guardrail_adapter = guardrail_adapter
        # TextAttack expects this attribute to exist:
        self.model = None

    def get_num_labels(self):
        return 2

    def __call__(self, text_input_list: List[str], **kwargs) -> np.ndarray:
        """
        Process a batch of text inputs.

        Args:
            text_input_list: List of text strings to classify

        Returns:
            Array of prediction probabilities [benign_prob, malicious_prob]
        """
        results = []

        for text in text_input_list:
            try:
                result: GuardResult = self.guardrail_adapter.check(text)

                # Convert to probability format expected by TextAttack
                if result.confidence is not None:
                    probs = GuardrailModelWrapper._risk_to_probs(result.confidence, mode="soft")
                else:
                    # fallback if no risk provided
                    probs = [0.9, 0.1] if result.allowed else [0.1, 0.9]

            except Exception as e:
                print(f"Error in guardrail prediction: {e}", file=sys.stderr)
                # Default to benign with low confidence
                probs = [0.5, 0.5]

            results.append(probs)
        return np.array(results)

    @staticmethod
    def _risk_to_probs(risk: float, mode="hard", tau=0.07) -> Tuple[float, float]:
        """
        Map risk∈[-1,1] to [p_safe, p_unsafe].  risk>0 => UNSAFE, risk<=0 => SAFE.
        mode="hard": match guard's decision exactly (argmax aligns 1:1).
        mode="soft": sigmoid around 0 if you want a smooth score.
        """
        if mode == "hard":
            # For LLM Guard: risk > 0 means injection detected (unsafe)
            p_unsafe = 0.99 if risk > 0 else 0.01
        else:  # "soft"
            # Sigmoid transformation for smooth probabilities
            p_unsafe = 1.0 / (1.0 + math.exp(-(risk / tau)))
            # Ensure risk=0 is slightly on the safe side
            if abs(risk) < 1e-8:
                p_unsafe = 0.49
        return 1.0 - p_unsafe, p_unsafe
