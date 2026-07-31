import pytest

from evalkit.batch import aggregate_results, run_queue
from ploshcha_sim.adapters.queue_sqlite import SqliteQueue
from ploshcha_sim.domain.governor import Governor
from ploshcha_sim.ports.queue import WorkItem


@pytest.fixture
def path(tmp_path):
    return tmp_path / "work.db"


def _queue(path, **kw):
    return SqliteQueue(path, **kw)


def test_the_same_key_twice_is_not_two_items(path):
    """Інгест мусить бути перезапускним із нуля: повтор — нормальний випадок, не помилка."""
    q = _queue(path)
    assert q.put("doc-1", {"text": "а"}) is True
    assert q.put("doc-1", {"text": "а"}) is False
    assert q.stats() == {"pending": 1}


def test_a_lease_is_not_a_handout(path):
    q = _queue(path)
    q.put("doc-1", {})
    item = q.lease("w1")
    assert item.key == "doc-1" and item.attempts == 1
    assert q.lease("w2") is None, "арендоване не видається другому воркеру"


def test_work_survives_a_dead_process(path):
    """★ Критерій готовності K8: перезапуск не губить і не дублює роботу.

    Емулюємо краш буквально: арендували, НЕ підтвердили, викинули обʼєкт черги й відкрили новий на
    тому самому файлі. Робота мусить бути на місці й повернутись у чергу після `recover_stale`.
    """
    q = _queue(path, clock=lambda: 1000.0)
    q.put("doc-1", {})
    q.put("doc-2", {})
    q.lease("w1")
    del q

    fresh = _queue(path, clock=lambda: 2000.0)
    assert fresh.stats() == {"leased": 1, "pending": 1}
    assert fresh.recover_stale(older_than_s=60) == 1
    assert fresh.stats() == {"pending": 2}
    keys = {fresh.lease("w2").key, fresh.lease("w2").key}
    assert keys == {"doc-1", "doc-2"}


def test_a_fresh_lease_is_not_stolen_by_recovery(path):
    now = [1000.0]
    q = _queue(path, clock=lambda: now[0])
    q.put("doc-1", {})
    q.lease("w1")
    now[0] = 1010.0
    assert q.recover_stale(older_than_s=300) == 0, "жива аренда не має відбиратись"


def test_a_broken_item_goes_dead_instead_of_eating_the_run(path):
    """Один зламаний документ не має крутитись вічно: інакше він зʼїдає бюджет прогону."""
    q = _queue(path, max_attempts=2)
    q.put("doc-1", {})
    for _ in range(2):
        item = q.lease("w1")
        q.fail(item.key, "boom")
    assert q.stats() == {"dead": 1}
    assert q.lease("w1") is None


def test_ack_records_the_result_for_aggregation(path):
    q = _queue(path)
    q.put("doc-1", {})
    q.ack("doc-1", {"findings": [{"claim": "х"}], "tokens": 10})
    done = q.results()
    assert len(done) == 1 and done[0].result["tokens"] == 10


def test_results_are_ordered_so_aggregation_is_deterministic(path):
    q = _queue(path)
    for key in ("doc-3", "doc-1", "doc-2"):
        q.put(key, {})
        q.ack(key, {"findings": []})
    assert [i.key for i in q.results()] == ["doc-1", "doc-2", "doc-3"]


# ── governor ────────────────────────────────────────────────────────────────────────────────────
def test_no_limit_means_no_limit_not_a_magic_number():
    assert Governor().can_continue(next_tokens=10**9) is True


def test_the_token_limit_is_checked_before_spending():
    """Перевірка ПІСЛЯ витрати означала б «ліміт плюс один айтем»."""
    gov = Governor(max_tokens=1000, reserve=0.0)
    gov.record(tokens=900)
    assert gov.can_continue(next_tokens=50) is True
    assert gov.can_continue(next_tokens=200) is False
    assert "межа токенів" in gov.stop_reason(next_tokens=200)


def test_the_reserve_keeps_the_last_item_inside_the_limit():
    gov = Governor(max_tokens=1000, reserve=0.1)
    gov.record(tokens=880)
    assert gov.can_continue(next_tokens=50) is False, "резерв тримає межу, а не переступає її"


def test_the_item_limit_counts_finished_work():
    gov = Governor(max_items=2)
    gov.record(tokens=1)
    assert gov.can_continue() is True
    gov.record(tokens=1)
    assert gov.can_continue() is False


def test_the_kill_switch_stops_regardless_of_budget(tmp_path):
    kill = tmp_path / "STOP"
    gov = Governor(max_tokens=10**9, kill_file=str(kill))
    assert gov.can_continue() is True
    kill.write_text("")
    assert gov.can_continue() is False and gov.stop_reason() == "kill-switch"


# ── пакетний прогін ─────────────────────────────────────────────────────────────────────────────
def test_the_batch_finishes_the_queue_and_reports(path):
    q = _queue(path)
    for i in range(5):
        q.put(f"doc-{i}", {"n": i})
    report = run_queue(q, "w1", lambda item: {"findings": [item.payload["n"]], "tokens": 10})
    assert report.done == 5 and report.tokens == 50
    assert q.stats() == {"done": 5}


def test_a_handler_crash_fails_one_item_not_the_run(path):
    q = _queue(path)
    for i in range(3):
        q.put(f"doc-{i}", {"n": i})

    def handler(item):
        if item.payload["n"] == 1:
            raise ValueError("зламаний документ")
        return {"findings": [item.payload["n"]]}

    report = run_queue(q, "w1", handler)
    assert report.done == 2 and report.failed >= 1
    assert "dead" in q.stats() or "pending" in q.stats()


def test_the_batch_stops_on_the_governor_not_on_the_queue(path):
    q = _queue(path)
    for i in range(10):
        q.put(f"doc-{i}", {})
    gov = Governor(max_items=3)
    report = run_queue(q, "w1", lambda item: {"findings": [1], "tokens": 5}, governor=gov)
    assert report.done == 3 and report.stopped == "межа айтемів 3"
    assert q.stats()["pending"] == 7, "решта лишається в черзі, а не гине"


def test_a_stopped_batch_resumes_where_it_left_off(path):
    q = _queue(path)
    for i in range(6):
        q.put(f"doc-{i}", {})
    handler = lambda item: {"findings": [1], "tokens": 5}  # noqa: E731
    run_queue(q, "w1", handler, governor=Governor(max_items=2))
    again = run_queue(q, "w1", handler, governor=Governor(max_items=10))
    assert again.done == 4 and q.stats() == {"done": 6}


def test_aggregation_names_the_dead_instead_of_hiding_them():
    """Прогін, що тихо загубив документи, гірший за той, що впав: перший ВИГЛЯДАЄ успішним."""
    items = [
        WorkItem(key="a", status="done", result={"findings": [{"claim": "х"}]}),
        WorkItem(key="b", status="done", result={"findings": []}),
        WorkItem(key="c", status="dead", error="boom"),
    ]
    out = aggregate_results(items)
    assert out["dead"] == ["c"] and out["empty"] == ["b"]
    assert len(out["findings"]) == 1 and out["coverage"] == 0.333
