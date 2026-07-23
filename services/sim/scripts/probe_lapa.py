"""Смоук-проба живого API Lapathoniia. Деталі — docs/sprints/S1-llm-act.md."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


load_env(ROOT / ".env")

from ploshcha_sim.adapters.llm_openai import OpenAICompatLlm  # noqa: E402
from ploshcha_sim.adapters.trace_jsonl import JsonlTrace  # noqa: E402
from ploshcha_sim.agents import act  # noqa: E402
from ploshcha_sim.domain import POI, AgentState, Persona, WorldState, action_json_schema  # noqa: E402


def small_world() -> WorldState:
    return WorldState(
        pois={
            "ploshcha": POI(id="ploshcha", name="Площа", kind="square"),
            "kuznya": POI(id="kuznya", name="Кузня", kind="forge"),
            "krynytsia": POI(id="krynytsia", name="Криниця", kind="well"),
        },
        agents={
            "koval": AgentState(
                id="koval",
                persona=Persona(role="koval", name="Остап", bio="Сільський коваль."),
                location="kuznya",
            ),
            "mati": AgentState(
                id="mati",
                persona=Persona(role="mati", name="Оксана", bio="Мати трьох дітей."),
                location="kuznya",
            ),
        },
    )


def main() -> int:
    key = os.environ.get("LAPA_API_KEY", "")
    base = os.environ.get("LAPA_BASE_URL", "")
    model = os.environ.get("LAPA_MODEL", "")
    if not key:
        print("НЕМА LAPA_API_KEY у .env")
        return 1
    print(f"model={model}\nbase_url={base}\nkey={key[:6]}…{key[-4:]}\n")

    # A. звичайна генерація — чи API взагалі живий
    print("── A. plain generate ──")
    try:
        r = OpenAICompatLlm(model=model, base_url=base, api_key=key).generate(
            "Одним реченням: що таке толока?", max_tokens=80
        )
        print(f"ok  {r.latency_ms}ms  in={r.usage.prompt_tokens} out={r.usage.completion_tokens}")
        print(f"    {r.text.strip()[:300]}")
    except Exception as e:
        print(f"FAIL {type(e).__name__}: {e}")
        return 1

    # B. act() у трьох режимах structured — мікро-експеримент Осі A
    print("\n── B. act() × structured_mode ──")
    trace_path = ROOT / "eval" / "traces" / "probe.jsonl"
    trace = JsonlTrace(trace_path)
    rows = []
    for mode in ("guided", "json_object", "none"):
        llm = OpenAICompatLlm(model=model, base_url=base, api_key=key, structured_mode=mode)
        world = small_world()
        try:
            res = act(
                world,
                "koval",
                llm,
                observations=["Оксана прийшла до кузні"],
                trace=trace,
                run_id="probe",
                ablation={"structured_mode": mode},
            )
            rows.append((mode, res.schema_valid, res.world_valid, res.reject_reason, res.raw_output.strip()))
            print(f"\n[{mode}] schema={res.schema_valid} world={res.world_valid} reason={res.reject_reason}")
            print(f"  raw: {res.raw_output.strip()[:220]}")
            print(f"  action: {res.action.model_dump()}  usage in={res.usage.prompt_tokens} out={res.usage.completion_tokens}")
        except Exception as e:
            rows.append((mode, None, None, f"{type(e).__name__}", str(e)[:200]))
            print(f"\n[{mode}] ERROR {type(e).__name__}: {str(e)[:200]}")

    print("\n── зведення ──")
    for mode, sv, wv, reason, _ in rows:
        print(f"  {mode:<12} schema_valid={sv}  world_valid={wv}  reason={reason}")

    print(f"\ntrace -> {trace_path}")
    last = trace_path.read_text(encoding="utf-8").strip().split("\n")[-1]
    print("поля траси:", ", ".join(sorted(json.loads(last).keys())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
