"""Cost-governor: скільки ще можна витратити й коли зупинитись негайно.

На тисячі документів найдорожча помилка — не поганий промпт, а **забутий ліміт**. Тому рішення
«починати наступний айтем чи ні» ухвалює чистий домен, а не місце виклику: інакше кожен новий
скрипт забуде свою перевірку по-своєму.

Kill-switch окремо від бюджету: бюджет вичерпується поступово, а kill-switch — це «стоп зараз»,
незалежно від залишку.
"""

from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_RESERVE = 0.05


class Spend(BaseModel):
    items_done: int = 0
    tokens: int = 0
    usd: float = 0.0


class Governor(BaseModel):
    """`max_*` = нуль означає «без ліміту»: явна відсутність межі краща за магічне число."""

    max_items: int = 0
    max_tokens: int = 0
    max_usd: float = 0.0
    kill_file: str | None = None
    reserve: float = DEFAULT_RESERVE
    spend: Spend = Field(default_factory=Spend)

    def killed(self) -> bool:
        return bool(self.kill_file) and Path(self.kill_file).exists()

    def record(self, *, tokens: int = 0, usd: float = 0.0, items: int = 1) -> None:
        self.spend.items_done += items
        self.spend.tokens += tokens
        self.spend.usd += usd

    def stop_reason(self, *, next_tokens: int = 0, next_usd: float = 0.0) -> str | None:
        """`None` = можна продовжувати. Резерв тримає останній айтем від виходу за межу.

        Перевіряємо ПЕРЕД витратою, з оцінкою наступного кроку: інакше межа завжди перетинається
        рівно на один айтем, і «ліміт» означає «ліміт плюс один».
        """
        if self.killed():
            return "kill-switch"
        if self.max_items and self.spend.items_done >= self.max_items:
            return f"межа айтемів {self.max_items}"
        if self.max_tokens and self.spend.tokens + next_tokens > self.max_tokens * (1 - self.reserve):
            return f"межа токенів {self.max_tokens}"
        if self.max_usd and self.spend.usd + next_usd > self.max_usd * (1 - self.reserve):
            return f"межа витрат ${self.max_usd}"
        return None

    def can_continue(self, *, next_tokens: int = 0, next_usd: float = 0.0) -> bool:
        return self.stop_reason(next_tokens=next_tokens, next_usd=next_usd) is None
