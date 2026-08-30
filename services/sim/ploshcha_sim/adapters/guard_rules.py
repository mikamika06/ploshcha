"""Правилова охорона: нуль LLM.

Детектор інʼєкцій, який сам кличе модель, можна зламати тією ж інʼєкцією. Тому тут регулярки й
структура — це і є причина, а не економія.
"""

from ..domain.injection import (
    Screening,
    Threat,
    blank_orders,
    screen,
    strip_threats,
    wrap_untrusted,
)
from ..ports.guard import GuardPort, Policy


class RuleGuard(GuardPort):
    def __init__(self, policy: Policy | None = None):
        self.policy = policy or Policy()

    def screen(self, text: str, *, tag: str | None = None) -> Screening:
        return screen(text, tag=tag)

    def cuts(self, text: str) -> list[Threat]:
        """Ніж і рахівник читають ОДНУ політику каналу — саме її, а не здогад того, хто питає.

        За `on_threat="note"` ніж не заведено, і той самий список називає наказ, який доїхав до
        промпта неушкодженим: інцидент потрібен саме тоді, тому політика різання тут не питається.
        """
        return strip_threats(text, spoken=self.policy.spoken)[1]

    def blanked(self, text: str) -> str:
        """Ніж і це зведення читають ОДНУ політику каналу — саме ту, з якою складено охорону."""
        return blank_orders(text, spoken=self.policy.spoken)

    def prepare(self, text: str, *, tag: str, trust: str = "untrusted") -> str:
        if trust == "trusted":
            return text
        body = text
        if self.policy.on_threat == "strip":
            body, _ = strip_threats(body, spoken=self.policy.spoken)
        if not self.policy.wrap_untrusted:
            return body
        return wrap_untrusted(body, tag=tag)
