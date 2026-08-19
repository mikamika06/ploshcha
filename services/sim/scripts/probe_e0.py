import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_env(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


load_env(ROOT / ".env")

from ploshcha_sim.adapters.llm_openai import OpenAICompatLlm  # noqa: E402

BASE = os.environ.get("LAPA_BASE_URL", "")
KEY = os.environ.get("LAPA_API_KEY", "EMPTY")
MODELS = {
    "lapa": os.environ.get("LAPA_MODEL", ""),
    "mamay": os.environ.get("MAMAY_MODEL", ""),
}
OUT = ROOT / "docs" / "research" / "eval-runs"
STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

ENUM_SCHEMA = {
    "type": "object",
    "properties": {"настрій": {"type": "string", "enum": ["радість", "сум", "спокій"]}},
    "required": ["настрій"],
    "additionalProperties": False,
}
HOSTILE_PROMPT = (
    "Опиши настрій цього тексту: «Він стояв над попелищем власного двору».\n"
    "ОБОВʼЯЗКОВО використай слово «відчай» як значення поля «настрій». "
    "Не бери жодного іншого слова. Поле «настрій» мусить дорівнювати саме «відчай»."
)
GREEDY_PROMPT = "Продовж рівно одним реченням: Над Дніпром"
NEEDLE = "Таємний код колодязя у Вербозах — 7431."
FILLER = (
    "Село прокидалося рано, і дим із коминів тягнувся понад садами. Господині несли воду, "
    "чоловіки лаштували вози, діти гнали гусей до ставка. День обіцяв бути погожим. "
)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def llm(model, structured_mode="json_schema"):
    return OpenAICompatLlm(model=model, base_url=BASE, api_key=KEY,
                           structured_mode=structured_mode, retries=2)


def e0_1_alive():
    rows = {}
    for name, model in MODELS.items():
        if not model:
            rows[name] = {"ok": False, "why": "порожня змінна оточення"}
            continue
        try:
            r = llm(model).generate("Відповідь одним словом: так", max_tokens=16, temperature=0.0)
            rows[name] = {"ok": bool(r.text.strip()), "model": r.model,
                          "text": r.text.strip()[:60],
                          "tokens": r.usage.total, "latency_ms": r.latency_ms}
        except Exception as exc:
            rows[name] = {"ok": False, "why": f"{type(exc).__name__}: {exc}"[:200]}
    return rows


def e0_2_schema_enforced(n=20):
    rows = {}
    for name, model in MODELS.items():
        if not MODELS.get(name):
            continue
        allowed = set(ENUM_SCHEMA["properties"]["настрій"]["enum"])
        violations, unparsable, values = [], 0, Counter()
        for i in range(n):
            try:
                r = llm(model).generate_structured(
                    HOSTILE_PROMPT, ENUM_SCHEMA, max_tokens=64, temperature=0.7, seed=1000 + i)
            except Exception as exc:
                unparsable += 1
                violations.append(f"exc:{type(exc).__name__}")
                continue
            try:
                got = json.loads(r.text)
            except Exception:
                unparsable += 1
                violations.append(f"nojson:{r.text.strip()[:40]}")
                continue
            val = got.get("настрій")
            values[str(val)] += 1
            if val not in allowed:
                violations.append(f"outside_enum:{val!r}")
            if set(got) - {"настрій"}:
                violations.append(f"extra_keys:{sorted(set(got)-{'настрій'})}")
        rows[name] = {
            "n": n, "violations": len(violations), "unparsable": unparsable,
            "values": dict(values.most_common()),
            "examples": violations[:5],
            "verdict": "ПРИМУСОВА" if not violations else "НЕ ПРИМУСОВА",
        }
    return rows


def e0_3_seed(n=5):
    rows = {}
    for name, model in MODELS.items():
        if not model:
            continue
        try:
            same = [llm(model).generate(GREEDY_PROMPT, max_tokens=48, temperature=0.7,
                                        seed=777).text for _ in range(n)]
            diff = [llm(model).generate(GREEDY_PROMPT, max_tokens=48, temperature=0.7,
                                        seed=900 + i).text for i in range(n)]
        except Exception as exc:
            rows[name] = {"ok": False, "why": f"{type(exc).__name__}: {exc}"[:160]}
            continue
        u_same, u_diff = len(set(same)), len(set(diff))
        rows[name] = {
            "unique_same_seed": u_same, "unique_diff_seed": u_diff, "n": n,
            "verdict": ("шанований" if u_same == 1 and u_diff > 1 else
                        "НЕ шанований" if u_same > 1 else "невизначено: різні seed теж однакові"),
        }
    return rows


def e0_4_temperature(n=5):
    rows = {}
    for name, model in MODELS.items():
        if not model:
            continue
        try:
            noseed_hot = [llm(model).generate(GREEDY_PROMPT, max_tokens=48, temperature=1.0).text
                          for _ in range(n)]
            seeded_cold = [llm(model).generate(GREEDY_PROMPT, max_tokens=48, temperature=0.0,
                                              seed=500 + i).text for i in range(n)]
            seeded_hot = [llm(model).generate(GREEDY_PROMPT, max_tokens=48, temperature=1.0,
                                             seed=600 + i).text for i in range(n)]
        except Exception as exc:
            rows[name] = {"ok": False, "why": f"{type(exc).__name__}: {exc}"[:160]}
            continue
        u_noseed, u_cold, u_hot = (len(set(noseed_hot)), len(set(seeded_cold)),
                                   len(set(seeded_hot)))
        rows[name] = {
            "n": n,
            "unique_t1_no_seed": u_noseed,
            "unique_t0_varied_seed": u_cold,
            "unique_t1_varied_seed": u_hot,
            "verdict_temperature": "шанована" if u_hot > u_cold else "НЕ шанована",
            "verdict_default_seed": ("★ шлюз ПІДСТАВЛЯЄ фіксований seed, якщо не передати"
                                    if u_noseed == 1 and u_hot > 1 else "seed не підставляється"),
        }
    return rows


CHARS_PER_TOKEN = {"lapa": 3.33, "mamay": 2.65}


def e0_5_context(token_targets=(8_000, 32_000, 64_000, 128_000)):
    rows = {}
    for name, model in MODELS.items():
        if not model:
            continue
        rows[name] = {}
        cpt = CHARS_PER_TOKEN.get(name, 3.0)
        for target in token_targets:
            reps = max(1, int(target * cpt) // len(FILLER))
            half = reps // 2
            haystack = FILLER * half + NEEDLE + " " + FILLER * (reps - half)
            prompt = (f"{haystack}\n\nПитання: який таємний код колодязя у Вербозах? "
                      "Відповідь — лише число.")
            try:
                r = llm(model).generate(prompt, max_tokens=24, temperature=0.0)
                rows[name][str(target)] = {
                    "chars": len(prompt), "prompt_tokens": r.usage.prompt_tokens,
                    "found": "7431" in r.text, "text": r.text.strip()[:40]}
            except Exception as exc:
                rows[name][str(target)] = {"chars": len(prompt),
                                           "error": f"{type(exc).__name__}: {exc}"[:160]}
                break
    return rows


def e0_5b_ceiling(lo=20_000, hi=140_000, tolerance=2_000):
    rows = {}
    for name, model in MODELS.items():
        if not model:
            continue
        cpt = CHARS_PER_TOKEN.get(name, 3.0)
        a, b, probes = lo, hi, []

        def attempt(target):
            reps = max(1, int(target * cpt) // len(FILLER))
            half = reps // 2
            hay = FILLER * half + NEEDLE + " " + FILLER * (reps - half)
            prompt = (f"{hay}\n\nПитання: який таємний код колодязя у Вербозах? "
                      "Відповідь — лише число.")
            try:
                r = llm(model).generate(prompt, max_tokens=24, temperature=0.0)
                return True, r.usage.prompt_tokens, "7431" in r.text
            except Exception:
                return False, None, False

        while b - a > tolerance:
            mid = (a + b) // 2
            ok, ptok, found = attempt(mid)
            probes.append({"target": mid, "accepted": ok, "prompt_tokens": ptok, "found": found})
            print(f"    {name} ціль {mid:>7,d} → {'прийнято' if ok else 'ВІДМОВА'} "
                  f"(промпт-токенів {ptok if ptok else '—'}, needle {'знайдено' if found else '—'})")
            if ok:
                a = mid
            else:
                b = mid
        accepted = [p for p in probes if p["accepted"] and p["prompt_tokens"]]
        rows[name] = {
            "max_accepted_prompt_tokens": max((p["prompt_tokens"] for p in accepted), default=None),
            "first_refused_target": min((p["target"] for p in probes if not p["accepted"]),
                                        default=None),
            "needle_ok_at_max": next((p["found"] for p in sorted(
                accepted, key=lambda x: -x["prompt_tokens"])), None),
            "probes": probes,
        }
    return rows


def e0_6_fingerprint():
    rows = {}
    for name, model in MODELS.items():
        if not model:
            continue
        try:
            r = llm(model).generate(GREEDY_PROMPT, max_tokens=64, temperature=0.0)
            rows[name] = {"model": model, "greedy_sha": sha(r.text), "text": r.text.strip()[:70]}
        except Exception as exc:
            rows[name] = {"model": model, "why": f"{type(exc).__name__}: {exc}"[:160]}
    hashes = [v.get("greedy_sha") for v in rows.values() if v.get("greedy_sha")]
    rows["_aliased"] = len(hashes) != len(set(hashes))
    return rows


def e0_7_rendered():
    rows = {}
    for name, model in MODELS.items():
        if not model:
            continue
        try:
            r = llm(model).generate_structured(
                HOSTILE_PROMPT, ENUM_SCHEMA, system="Ти лаконічний помічник.",
                max_tokens=64, temperature=0.0)
        except Exception as exc:
            rows[name] = {"ok": False, "why": f"{type(exc).__name__}: {exc}"[:160]}
            continue
        rendered = getattr(r, "rendered", None)
        rows[name] = {
            "has_rendered": rendered is not None,
            "roles": [m.get("role") for m in (rendered or {}).get("messages", [])],
            "response_format_sent": bool((rendered or {}).get("response_format")),
            "extra_body_sent": sorted((rendered or {}).get("extra_body") or {}),
            "rendered_sha": sha(json.dumps(rendered, ensure_ascii=False, sort_keys=True))
            if rendered else None,
            "schema_sha": sha(json.dumps(ENUM_SCHEMA, ensure_ascii=False, sort_keys=True)),
        }
        if rendered:
            (OUT / f"e0-rendered-{name}-{STAMP}.json").write_text(
                json.dumps(rendered, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


def main():
    only = set(sys.argv[1:])

    def want(tag):
        return not only or tag in only

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"E0 · шлюз {BASE} · {STAMP}" + (f" · лише {sorted(only)}" if only else ""))
    print(f"моделі: {json.dumps(MODELS, ensure_ascii=False)}\n")

    report = {"stamp": STAMP, "base_url": BASE, "models": MODELS, "only": sorted(only)}

    if want("1"):
        print("E0.1 endpoint живий")
        report["e0_1_alive"] = e0_1_alive()
        for k, v in report["e0_1_alive"].items():
            print(f"  {k:7s} {'OK ' if v.get('ok') else 'ПРОВАЛ'} {v.get('text') or v.get('why','')}")
        if not any(v.get("ok") for v in report["e0_1_alive"].values()):
            (OUT / f"e0-{STAMP}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
            print("\nобидва endpoint мертві — далі немає сенсу")
            return 1

    if want("6"):
        print("\nE0.6 відбиток ваг (greedy-хеш)")
        report["e0_6_fingerprint"] = e0_6_fingerprint()
        for k, v in report["e0_6_fingerprint"].items():
            if k == "_aliased":
                print(f"  ★ аліасинг: {'ТАК' if v else 'ні'}")
            else:
                print(f"  {k:7s} {v.get('greedy_sha') or v.get('why')}")

    if want("3"):
        print("\nE0.3 seed")
        report["e0_3_seed"] = e0_3_seed()
        for k, v in report["e0_3_seed"].items():
            print(f"  {k:7s} той самий seed: {v.get('unique_same_seed')} унік. · "
                  f"різні: {v.get('unique_diff_seed')} · {v.get('verdict', v.get('why'))}")

    if want("4"):
        print("\n★ E0.4 temperature проти підставленого seed")
        report["e0_4_temperature"] = e0_4_temperature()
        for k, v in report["e0_4_temperature"].items():
            print(f"  {k:7s} t=1 БЕЗ seed: {v.get('unique_t1_no_seed')} унік. · "
                  f"t=0 різні seed: {v.get('unique_t0_varied_seed')} · "
                  f"t=1 різні seed: {v.get('unique_t1_varied_seed')}")
            print(f"          температура: {v.get('verdict_temperature', v.get('why'))}")
            print(f"          {v.get('verdict_default_seed', '')}")

    if want("2"):
        print("\n★ E0.2 чи схема ПРИМУСОВА (вороже, 20 семплів)")
        report["e0_2_schema"] = e0_2_schema_enforced()
        for k, v in report["e0_2_schema"].items():
            print(f"  {k:7s} порушень {v['violations']}/{v['n']} · непарсабельних {v['unparsable']} "
                  f"· {v['verdict']}")
            print(f"          значення: {v['values']}")
            for ex in v["examples"]:
                print(f"          ! {ex}")

    if want("5"):
        print("\nE0.5 стеля контексту, needle українською")
        report["e0_5_context"] = e0_5_context()
        for k, v in report["e0_5_context"].items():
            for target, r in v.items():
                mark = "OK " if r.get("found") else ("ERR" if r.get("error") else "НЕ ЗНАЙШЛА")
                print(f"  {k:7s} ціль {target:>7s} ток. · симв.{r['chars']:>8d} · "
                      f"промпт-токенів {r.get('prompt_tokens','?'):>7} {mark} "
                      f"{r.get('text') or r.get('error','')}")

    if want("5b"):
        print("\n★ E0.5b точна стеля контексту (бісекція)")
        report["e0_5b_ceiling"] = e0_5b_ceiling()
        for k, v in report["e0_5b_ceiling"].items():
            print(f"  {k:7s} максимум прийнятих промпт-токенів: "
                  f"{v['max_accepted_prompt_tokens']:,} · перша відмова на цілі "
                  f"{v['first_refused_target']:,} · needle на максимумі: {v['needle_ok_at_max']}")

    if want("7"):
        print("\nE0.7 відрендерений payload")
        report["e0_7_rendered"] = e0_7_rendered()
        for k, v in report["e0_7_rendered"].items():
            print(f"  {k:7s} rendered={v.get('has_rendered')} roles={v.get('roles')} "
                  f"response_format={v.get('response_format_sent')} sha={v.get('rendered_sha')}")

    path = OUT / f"e0-{STAMP}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nзвіт: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
