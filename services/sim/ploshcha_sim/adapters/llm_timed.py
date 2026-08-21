"""Обгортка, що міряє ТРИВАЛІСТЬ виклику моделі.

Питання «чому так довго» має отримувати число, а не здогад. Тривалість уже приходить у
`LlmResult.latency_ms`, але до сервера вона не доїжджала: у трасу потрапляють лише озвучені
виклики, а найдовший — планування партитури — не озвучується взагалі.

Обгортка нічого не змінює у відповіді: вона лише доповідає, скільки виклик тривав.
"""

from ..ports.llm import LlmPort, LlmResult


class TimedLlm(LlmPort):
    def __init__(self, inner: LlmPort, sink, label: str):
        self._inner = inner
        self._sink = sink
        self._label = label

    @property
    def model(self) -> str:
        return self._inner.model

    def generate(self, *args, **kwargs) -> LlmResult:
        return self._note(self._inner.generate(*args, **kwargs))

    def generate_structured(self, *args, **kwargs) -> LlmResult:
        return self._note(self._inner.generate_structured(*args, **kwargs))

    def _note(self, res: LlmResult) -> LlmResult:
        try:
            self._sink.note_latency(self._label, res.latency_ms)
        except Exception:
            pass  # вимір не має права зламати прогін
        return res


class LatencyBook:
    """Останні заміри тривалості по ярусах. У памʼяті: це прилад, а не журнал."""

    KEEP = 40

    def __init__(self) -> None:
        self._rows: dict[str, list[int]] = {}

    def note_latency(self, label: str, ms: int) -> None:
        row = self._rows.setdefault(label, [])
        row.append(int(ms))
        if len(row) > self.KEEP:
            del row[: len(row) - self.KEEP]

    def summary(self) -> dict:
        out = {}
        for label, row in self._rows.items():
            if not row:
                continue
            ordered = sorted(row)
            out[label] = {"n": len(row), "median_ms": ordered[len(ordered) // 2],
                          "max_ms": ordered[-1]}
        return out
