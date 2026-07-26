import json

from pydantic import BaseModel

from ..ports.router import EffortPolicy, ModelRouter
from ..ports.trace import StepRecord, TracePort

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"accepted": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["accepted", "reason"],
    "additionalProperties": False,
}

SYSTEM = (
    "Ти суворий перевіряльник. Спробуй СПРОСТУВАТИ відповідь на задачу. "
    "Якщо спростувати не вдається — accepted=true, інакше false. За сумніву — false. "
    "Відповідай РІВНО одним JSON: {\"accepted\":<bool>,\"reason\":\"<чому>\"}"
)


class Verdict(BaseModel):
    accepted: bool
    reason: str = ""


def verify(task, answer, router: ModelRouter, effort: EffortPolicy, *,
           seed: int = 0, trace: TracePort | None = None, run_id: str = "") -> Verdict:
    llm = router.route("judge")
    cfg = effort.effort("judge")
    prompt = f"Задача: {task}\nВідповідь: {answer}\n\nСпробуй спростувати. Один JSON."
    res = llm.generate_structured(prompt, VERDICT_SCHEMA, system=SYSTEM, max_tokens=cfg.max_tokens, seed=seed)
    try:
        d = json.loads(res.text)
        verdict = Verdict(accepted=bool(d["accepted"]), reason=str(d.get("reason", "")))
        ok = True
    except (json.JSONDecodeError, KeyError, TypeError):
        verdict = Verdict(accepted=False, reason="verify_parse_fail")
        ok = False
    if trace is not None:
        trace.emit(StepRecord(
            run_id=run_id, tick=0, agent="verifier", stage="judge", model=llm.model,
            prompt=prompt, raw_output=res.text, parsed=verdict.model_dump(),
            schema_valid=ok, world_valid=ok, reject_reason=None if ok else "verify_parse_fail",
            usage=res.usage, latency_ms=res.latency_ms, finish_reason=res.finish_reason, seed=seed,
        ))
    return verdict
