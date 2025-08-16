from guard_adapters.guardrail_adapter import GuardrailAdapter


class DefenseAmplifier:
    """
    Implementation of the defense amplifier.
    """
    def __init__(self, guardrail: GuardrailAdapter):
        self.attacker = None
        self.guardrail = guardrail

        self.attacker = self._get_attacker()

    def _get_attacker(self):
        """Lazy import to avoid circular dependency"""
        if self.attacker is None:
            from attacker.aml_attacker import AmlAttacker
            self.attacker = AmlAttacker()
        return self.attacker

    def apply_defense(self, text: str) -> bool:
        """
        Apply defense will return an enhanced response of if the text is malicious or not.
        :param text: the input text for the guardrail.
        :return: True if the text is valid, False otherwise.
        """
        from attacker.aml_attacker import AMLTechnique

        attack_results = self.attacker.attack([(text, 0)], AMLTechnique.TEXTFOOLER, self.guardrail, 10, True)

        # If we succeeded to change the guardrail classification via minor changes to the input that don't change
        # semantic meaning, mark the input as invalid and return False.
        if len(attack_results) > 0:
            return False
        return True

