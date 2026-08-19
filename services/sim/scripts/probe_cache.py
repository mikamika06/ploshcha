import json
import os
import sys
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
MODELS = {"lapa": os.environ.get("LAPA_MODEL", ""), "mamay": os.environ.get("MAMAY_MODEL", "")}

FILLER = ("Село прокидалося рано, і дим із коминів тягнувся понад садами. Господині несли воду, "
          "чоловіки лаштували вози, діти гнали гусей до ставка. ")
PREFIX = FILLER * 120
ROUNDS = 3


def discriminate(name, llm):
    base = PREFIX + "\n\nОдним словом: яка пора дня описана?"
    a = llm.generate(base, max_tokens=16, temperature=0.7, seed=99)
    b = llm.generate(base + " ", max_tokens=16, temperature=0.7, seed=99)
    c = llm.generate(PREFIX + "\n\nОдним словом: яку тварину згадано?",
                     max_tokens=16, temperature=0.0, seed=99)
    identical_after_nonce = a.text == b.text
    print(f"  {name:7s} той самий запит + пробіл, той самий seed:")
    print(f"          A={a.text.strip()[:40]!r} ({a.latency_ms} мс)")
    print(f"          B={b.text.strip()[:40]!r} ({b.latency_ms} мс)")
    print(f"          → однакові: {identical_after_nonce} "
          f"{'★ SEED справжній' if identical_after_nonce else '★ це був КЕШ ВІДПОВІДІ'}")
    print(f"  {name:7s} той самий ПРЕФІКС, інше питання: {c.latency_ms} мс "
          f"(промпт-токенів {c.usage.prompt_tokens})")
    return {"nonce_identical": identical_after_nonce,
            "latency_nonce_ms": b.latency_ms, "latency_first_ms": a.latency_ms,
            "latency_same_prefix_new_suffix_ms": c.latency_ms,
            "text_a": a.text.strip()[:80], "text_b": b.text.strip()[:80]}


def main():
    print(f"L10 · перевірка кешу префікса · шлюз {BASE}\n")
    report = {}
    for name, model in MODELS.items():
        if not model:
            continue
        llm = OpenAICompatLlm(model=model, base_url=BASE, api_key=KEY, retries=2)
        rows = []
        for i in range(ROUNDS):
            r = llm.generate(PREFIX + "\n\nОдним словом: яка пора дня описана?",
                             max_tokens=16, temperature=0.0, seed=42)
            rows.append({"round": i + 1, "prompt_tokens": r.usage.prompt_tokens,
                         "completion": r.usage.completion_tokens, "latency_ms": r.latency_ms})
            print(f"  {name:7s} спроба {i+1}: промпт-токенів {r.usage.prompt_tokens:>6} · "
                  f"латентність {r.latency_ms:>6} мс")
        first, rest = rows[0]["latency_ms"], [x["latency_ms"] for x in rows[1:]]
        same_tokens = len({x["prompt_tokens"] for x in rows}) == 1
        speedup = first / (sum(rest) / len(rest)) if rest and sum(rest) else 0
        verdict = ("кеш ІМОВІРНО є" if speedup >= 1.5 and same_tokens
                   else "кешу НЕ ВИДНО")
        print(f"  {name:7s} → прискорення {speedup:.2f}× · токени однакові: {same_tokens} "
              f"· {verdict}\n")
        report[name] = {"rows": rows, "speedup": speedup, "same_prompt_tokens": same_tokens,
                        "verdict": verdict}
        report[name]["discriminate"] = discriminate(name, llm)
        print()
    out = ROOT / "docs" / "research" / "eval-runs" / "l10-cache.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"звіт: {out}")


if __name__ == "__main__":
    sys.exit(main())
