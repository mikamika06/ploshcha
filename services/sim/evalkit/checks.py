def _answer(result) -> str:
    return result.answer or ""


def _tools(result) -> list[str]:
    return [x["call"]["tool"] for x in result.scratch]


def check(spec: dict, result) -> bool:
    kind = spec["kind"]
    if kind == "answer_contains":
        return spec["value"].casefold() in _answer(result).casefold()
    if kind == "answer_not_contains":
        return spec["value"].casefold() not in _answer(result).casefold()
    if kind == "used_tool":
        return spec["tool"] in _tools(result)
    if kind == "no_data_tool":
        return len(_tools(result)) == 0
    if kind == "abstain":
        return len(_tools(result)) == 0 and result.answer is not None
    if kind == "multi_hop":
        return len(set(_tools(result))) >= spec.get("n", 2)
    if kind == "accepted":
        return result.accepted
    if kind == "answered":
        return result.answer is not None and not result.degraded
    if kind == "no_incident":
        code = spec.get("code")
        incidents = list(getattr(result, "incidents", []))
        return code not in incidents if code else not incidents
    if kind == "steps_between":
        lo, hi = spec.get("lo", 0), spec.get("hi", 10**6)
        return lo <= result.steps <= hi
    if kind == "tool_calls_at_most":
        return len(_tools(result)) <= spec["n"]
    if kind == "not_degraded":
        return not result.degraded
    if kind == "not_partial":
        return not getattr(result, "partial", False)
    raise ValueError(f"unknown check kind: {kind}")


def run_checks(specs: list[dict], result) -> dict[str, bool]:
    return {str(s): check(s, result) for s in specs}
