"""Одиниця результату суб-агента й правила злиття — чистий домен, без моделі.

Головне тут не структура, а один інваріант, знайдений у K9: **стан доказів дитини мусить підніматись
у синтез**. Якщо суб-агент повернув «даних немає», а супервізор бачить лише текст його відповіді, то
відсутність зникає — і супервізор латає дірку вигадкою, бо ніщо не сказало йому, що дірка є.

Тому `Finding` несе `evidence` тризначним, а `merge_evidence` зводить дитячі стани за тим самим
правилом, що й `evidence_state` для одного прогону.
"""

from typing import Literal

from pydantic import BaseModel, Field

from .evidence import Outcome

FindingKind = Literal["answer", "abstain", "failure"]


class Finding(BaseModel):
    """Те, що дитина віддає батькові. `provenance` — хто саме, щоб траса лишалась ієрархічною."""

    task: str
    claim: str | None = None
    kind: FindingKind = "answer"
    evidence: bool | None = None
    confidence: float | None = None
    provenance: str = ""
    tokens: int = 0
    steps: int = 0
    # Ярусна розкладка дитини піднімається як є. Вигаданий ярус «child» тарифікувався за
    # UNKNOWN_PRICE (найдорожча ставка) і з нульовою часткою промпту, тому гроші графа були
    # завищені: при ширині 4 відношення в токенах 2.16×, а в «доларах» виходило 2.59×.
    tokens_by_lane: dict[str, int] = Field(default_factory=dict)
    prompt_by_lane: dict[str, int] = Field(default_factory=dict)
    incidents: list[str] = Field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.kind == "answer" and bool((self.claim or "").strip())


def merge_evidence(findings: list[Finding]) -> bool | None:
    """True — хоч щось знайдено; False — шукали й НЕ знайшли; None — пошуку не було.

    Те саме правило, що `evidence_state` для одного прогону: підйом на рівень вище не має його
    змінювати, інакше «даних немає» перетворюється на «не знаю, чи є».
    """
    seen_absent = False
    for finding in findings:
        if finding.evidence is True:
            return True
        if finding.evidence is False:
            seen_absent = True
    return False if seen_absent else None


def merge_outcome(findings: list[Finding]) -> Outcome:
    """Часткова відповідь — це відповідь; повна відсутність — відмова; порожнеча — збій.

    Порядок перевірок навмисний: спершу дивимось, чи є хоч одна ЗМІСТОВНА дитина, бо одна знайдена
    відповідь важливіша за дві відмови — інакше супервізор мовчав би там, де має що сказати.
    """
    if not findings:
        return "failure"
    if any(f.usable for f in findings):
        return "answer"
    if all(f.kind == "abstain" for f in findings):
        return "abstain"
    return "failure"


def missing_tasks(findings: list[Finding]) -> list[str]:
    """Про що саме дитина нічого не дізналась — щоб синтез називав дірку, а не приховував її."""
    return [f.task for f in findings if not f.usable]
