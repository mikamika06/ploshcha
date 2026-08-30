"""Скриптована модель для тестів — без мережі."""

from ..ports.llm import LlmPort, LlmResult, LlmUsage


class FakeLlm(LlmPort):
    def __init__(self, responses: list[str], model: str = "fake", finish_reason: str = "stop",
                 strict: bool = False):
        self.model = model
        self.finish_reason = finish_reason
        self.strict = strict
        self._responses = list(responses)
        self.calls: list[dict] = []
        # ★ Штраф повторення лежить ПОЛЕМ, а не аргументом `_next`, і це не лінощі.
        #
        # `_next` перевизначають девʼять фейків у тестах із жорсткою сигнатурою; новий аргумент
        # мовчки поламав би кожен із них, тобто ціна одного важеля була б девʼять чужих правок.
        # Поле дає те саме: у `calls` видно, з яким штрафом прийшов виклик.
        self._penalty: float | None = None

    def _next(self, prompt, system, structured, schema, seed, temperature=0.0,
              max_tokens=0) -> LlmResult:
        self.calls.append(
            {"prompt": prompt, "system": system, "structured": structured, "schema": schema,
             "seed": seed, "temperature": temperature, "max_tokens": max_tokens,
             "repetition_penalty": self._penalty}
        )
        if not self._responses and self.strict:
            raise AssertionError(f"FakeLlm: скрипт вичерпано на виклику #{len(self.calls)}")
        text = self._responses.pop(0) if self._responses else ""
        return LlmResult(
            text=text,
            model=self.model,
            usage=LlmUsage(prompt_tokens=len(prompt.split()), completion_tokens=len(text.split())),
            latency_ms=0,
            structured=structured,
            finish_reason=self.finish_reason,
        )

    def generate(self, prompt, *, system=None, temperature=0.0, max_tokens=512, seed=None,
                 repetition_penalty=None) -> LlmResult:
        self._penalty = repetition_penalty
        return self._next(prompt, system, False, None, seed, temperature, max_tokens)

    def generate_structured(self, prompt, schema, *, system=None, temperature=0.0, max_tokens=512,
                            seed=None, repetition_penalty=None) -> LlmResult:
        self._penalty = repetition_penalty
        return self._next(prompt, system, True, schema, seed, temperature, max_tokens)
