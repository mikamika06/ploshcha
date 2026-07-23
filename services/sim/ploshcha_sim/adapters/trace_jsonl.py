"""Траси: у памʼять (тести) і в JSON-lines файл (прогони)."""

from pathlib import Path

from ..ports.trace import StepRecord, TracePort


class InMemoryTrace(TracePort):
    def __init__(self) -> None:
        self.records: list[StepRecord] = []

    def emit(self, record: StepRecord) -> None:
        record.seq = len(self.records)
        self.records.append(record)


class JsonlTrace(TracePort):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0

    def emit(self, record: StepRecord) -> None:
        record.seq = self._seq
        self._seq += 1
        with self.path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
