"""Правилова охорона: нуль LLM.

Детектор інʼєкцій, який сам кличе модель, можна зламати тією ж інʼєкцією. Тому тут регулярки й
структура — це і є причина, а не економія.
"""

from ..domain.injection import Screening, screen, strip_threats, wrap_untrusted
from ..ports.guard import GuardPort, Policy


class RuleGuard(GuardPort):
    def __init__(self, policy: Policy | None = None):
        self.policy = policy or Policy()

    def screen(self, text: str, *, tag: str | None = None) -> Screening:
        return screen(text, tag=tag)

    def prepare(self, text: str, *, tag: str, trust: str = "untrusted") -> str:
        if trust == "trusted":
            return text
        body = text
        if self.policy.on_threat == "strip":
            body, _ = strip_threats(body)
        if not self.policy.wrap_untrusted:
            return body
        return wrap_untrusted(body, tag=tag)
