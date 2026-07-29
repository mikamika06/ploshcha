from evalkit.harness import EvalResult, load_items
from evalkit.prompts import REGISTRY
from evalkit.report import aggregate, paired


def _row(item="a", cond="base", seed=1, ok=True, checks=None, **kw):
    return EvalResult(item_id=item, category=kw.pop("category", "x"), condition=cond, seed=seed,
                      success=ok, checks=checks if checks is not None else {"c": ok}, **kw)


def test_progress_rate_gives_partial_credit():
    rows = [_row(ok=False, checks={"a": True, "b": True, "c": False, "d": False})]
    rep = aggregate(rows)[0]
    assert rep["success_rate"] == 0.0
    assert rep["progress_rate"] == 0.5, "половина предикатів пройшла"


def test_steps_to_answer_counts_only_accepted_non_partial():
    rows = [
        _row(item="a", ok=True, accepted=True, steps=2),
        _row(item="b", ok=False, accepted=False, steps=8),
        _row(item="c", ok=True, accepted=True, partial=True, steps=9),
    ]
    rep = aggregate(rows)[0]
    assert rep["steps_to_answer"] == 2.0, "провали й часткові не занижують метрику"
    assert rep["avg_steps"] > 2.0


def test_incident_profile_and_partial_counted():
    rows = [
        _row(item="a", incidents=["dup_call", "tool_error"]),
        _row(item="b", incidents=["dup_call"], partial=True),
    ]
    rep = aggregate(rows)[0]
    assert rep["incident_profile"] == {"dup_call": 2, "tool_error": 1}
    assert rep["partial"] == 1


def test_aux_tokens_reported_separately():
    rows = [_row(tokens=400, aux_tokens=270)]
    rep = aggregate(rows)[0]
    assert rep["avg_tokens"] == 400.0 and rep["avg_aux_tokens"] == 270.0


def test_paired_compares_same_item_and_seed():
    rows = [
        _row(item="a", cond="base", seed=1, ok=False, incidents=["dup_call"]),
        _row(item="a", cond="rec", seed=1, ok=True, incidents=["dup_call"]),
        _row(item="b", cond="base", seed=1, ok=True),
        _row(item="b", cond="rec", seed=1, ok=False),
        _row(item="c", cond="base", seed=1, ok=True),
        _row(item="c", cond="rec", seed=1, ok=True),
    ]
    p = paired(rows, "base", "rec")
    assert p["cells"] == 3
    assert p["fixed"] == 1 and p["broke"] == 1 and p["unchanged"] == 1
    assert p["net"] == 0
    assert p["rescued_with_incident"] == 1


def test_paired_ignores_cells_present_in_only_one_condition():
    rows = [
        _row(item="a", cond="base", seed=1, ok=True),
        _row(item="b", cond="rec", seed=1, ok=True),
    ]
    assert paired(rows, "base", "rec")["cells"] == 0


def test_recover_set_loads_and_covers_the_failure_classes():
    items = load_items(str(REGISTRY.parents[1] / "items" / "recover.jsonl"))
    assert len(items) >= 16
    cats = {i.category for i in items}
    assert cats >= {"multi_hop", "arg_variants", "tool_unknown", "tool_error", "abstain",
                    "budget", "grounding"}


def test_recover_set_uses_the_new_predicates():
    items = load_items(str(REGISTRY.parents[1] / "items" / "recover.jsonl"))
    kinds = {c["kind"] for i in items for c in i.checks}
    assert {"steps_between", "no_incident", "tool_calls_at_most", "not_partial"} <= kinds


def test_recover_chains_demand_more_than_two_steps():
    items = load_items(str(REGISTRY.parents[1] / "items" / "recover.jsonl"))
    chains = [i for i in items if i.id.startswith("chain-")]
    assert len(chains) >= 3
    for item in chains:
        lo = next(c["lo"] for c in item.checks if c["kind"] == "steps_between")
        assert lo >= 3, f"{item.id}: горизонт мусить бути довшим за 2 кроки"
