import pytest

from ploshcha_sim.agents.orchestrator import (
    VERBATIM_WINDOW,
    Orchestrator,
    _digest_line,
    _render,
)
from ploshcha_sim.domain.task import TaskState


def _state(n: int) -> TaskState:
    s = TaskState(task="Порахуй суми записів")
    for i in range(n):
        s.scratch.append({
            "call": {"tool": "запис", "ідентифікатор": f"сх-{1898 + i}-01"},
            "result": {"відомо": True, "ідентифікатор": f"сх-{1898 + i}-01",
                       "село": "Сухий Яр", "рік": 1898 + i, "майстер": "Остап Крученюк",
                       "сума": 29 + i, "примітка": "довгий текст " * 6},
        })
    return s


def test_window_none_keeps_whole_history():
    s = _state(6)
    text = _render(s, verbatim=None)
    for i in range(6):
        assert f"сх-{1898 + i}-01" in text


def test_window_two_keeps_only_last_two_verbatim():
    s = _state(6)
    text = _render(s, verbatim=2)
    assert text.count("Результат:") == 2
    assert "сх-1902-01" in text and "сх-1903-01" in text
    assert "майстер" not in text.split("Результат:")[0]


def test_window_shrinks_the_prompt_a_lot():
    s = _state(8)
    full = len(_render(s, verbatim=None))
    win = len(_render(s, verbatim=2))
    assert win < full * 0.45, f"вікно 2 дало {win} проти {full}"


def test_digest_keeps_identifiers_of_older_steps():
    s = _state(6)
    win = _render(s, verbatim=2)
    dig = _render(s, verbatim=2, digest=True)
    assert "сх-1898-01" not in win
    assert "сх-1898-01" in dig
    assert "Уже зроблено раніше:" in dig


def test_digest_is_cheaper_than_full_history():
    s = _state(8)
    full = len(_render(s, verbatim=None))
    dig = len(_render(s, verbatim=2, digest=True))
    assert dig < full, f"зведення {dig} мусить бути дешевше за повну історію {full}"


def test_digest_line_truncates_long_results():
    line = _digest_line(_state(1).scratch[0])
    assert line.startswith("  запис(ідентифікатор=сх-1898-01)")
    assert line.endswith("…")
    assert len(line) < 160


def test_digest_without_window_adds_nothing():
    s = _state(4)
    assert _render(s, verbatim=None, digest=True) == _render(s, verbatim=None)


class _Router:
    def route(self, kind):
        raise AssertionError("не має викликатись")

    def lane(self, kind):
        return "lapa"


class _Effort:
    def effort(self, kind):
        raise AssertionError("не має викликатись")


class _Tools:
    def specs(self):
        return []


@pytest.mark.parametrize("window,notebook,expected", [
    (None, None, None),
    (None, object(), VERBATIM_WINDOW),
    (2, None, 2),
    (2, object(), 2),
    (0, object(), 0),
])
def test_window_precedence_spec_over_notebook(window, notebook, expected):
    orch = Orchestrator(_Router(), _Effort(), _Tools(), history_window=window)
    assert orch._window(notebook) == expected


def test_spec_axis_defaults_do_not_move_existing_conditions():
    from evalkit.conditions import CONDITIONS

    base = CONDITIONS["hetero-plan@8"]
    assert base.history_window is None
    assert base.history_digest is False
    assert base.with_(history_window=2).sha256 != base.sha256
