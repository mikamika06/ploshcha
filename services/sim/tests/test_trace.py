"""Траси: jsonl-запис і послідовність."""

import json

from ploshcha_sim.adapters import JsonlTrace
from ploshcha_sim.ports import StepRecord


def _rec(agent: str) -> StepRecord:
    return StepRecord(
        run_id="r1",
        tick=0,
        agent=agent,
        stage="act",
        model="fake",
        prompt="p",
        raw_output="{}",
    )


def test_jsonl_writes_one_line_per_record(tmp_path):
    path = tmp_path / "traces" / "run.jsonl"
    t = JsonlTrace(path)
    t.emit(_rec("koval"))
    t.emit(_rec("mati"))

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["agent"] == "koval" and first["seq"] == 0
    assert json.loads(lines[1])["seq"] == 1


def test_jsonl_creates_parent_dir(tmp_path):
    JsonlTrace(tmp_path / "deep" / "nested" / "run.jsonl")
    assert (tmp_path / "deep" / "nested").is_dir()


def test_record_roundtrips_through_json(tmp_path):
    path = tmp_path / "run.jsonl"
    t = JsonlTrace(path)
    r = _rec("did")
    r.parsed = {"type": "wait"}
    r.schema_valid = True
    t.emit(r)
    back = StepRecord.model_validate_json(path.read_text(encoding="utf-8").strip())
    assert back.parsed == {"type": "wait"} and back.schema_valid
