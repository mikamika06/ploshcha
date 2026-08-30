"""Порт LLM."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class LlmUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LlmResult(BaseModel):
    text: str
    model: str
    usage: LlmUsage = Field(default_factory=LlmUsage)
    latency_ms: int = 0
    structured: bool = False
    finish_reason: str | None = None
    rendered: dict | None = None


class LlmPort(ABC):
    """Усі моделі (Lapa/Mamay/Gemma/GPT) ховаються за цим інтерфейсом."""

    model: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        repetition_penalty: float | None = None,
    ) -> LlmResult: ...

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: dict,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        repetition_penalty: float | None = None,
    ) -> LlmResult:
        """Вивід обмежений JSON-схемою (constrained decoding). Парсинг — на боці агента,
        щоб невалідний вивід лишався вимірюваним, а не ховався в порті.

        `repetition_penalty` — третій важіль декодування поруч із `temperature` і `max_tokens`, і
        стоїть він у порті з тієї самої причини, що й вони: це поле ЗАПИТУ, а не рішення агента.
        `None` означає «не слати поля взагалі» — байт-у-байт той самий запит, що й доти, тож усі
        вже пораховані прогони лишаються порівнюваними. Чому важіль існує й чому вимкнений —
        `OpenAICompatLlm._call`."""
