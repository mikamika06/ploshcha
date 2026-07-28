import hashlib
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

from ploshcha_sim.adapters.llm_openai import OpenAICompatLlm

PROMPTS = [
    "Продовж одним реченням: Гетьман Іван Мазепа був",
    "Назви три українські народні свята зимового циклу.",
    "Обчисли 347 * 892 і напиши лише число.",
    "Continue in one sentence: The capital of Ukraine is",
]


def probe(llm) -> tuple[str, list[str]]:
    parts = []
    for p in PROMPTS:
        text = llm.generate(p, temperature=0.0, max_tokens=48, seed=1).text
        parts.append(hashlib.sha256(text.encode()).hexdigest()[:8])
    joint = hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]
    return joint, parts


def parse_targets(argv) -> list[tuple[str, str, str, str]]:
    key, url = os.environ.get("LAPA_API_KEY", "EMPTY"), os.environ.get("LAPA_BASE_URL", "")
    targets = [(f"gw:{m}", m, url, key) for m in (
        "LapaLLM-Gemma-3-12B-instruct",
        "MamayLM-Gemma-3-12B-IT-v1.0",
        "MamayLM-Gemma-3-27B-IT-v2.0",
    )] if url else []
    for a in argv:
        if a.startswith("--local="):
            label, port = a.split("=", 1)[1].split(":")
            targets.append((f"local:{label}", label, f"http://127.0.0.1:{port}/v1", "EMPTY"))
    return targets


def main():
    targets = parse_targets(sys.argv[1:])
    rows = []
    for label, model, url, key in targets:
        try:
            llm = OpenAICompatLlm(model=model, base_url=url, api_key=key, timeout=300.0)
            joint, parts = probe(llm)
            rows.append({"target": label, "joint": joint, "per_prompt": parts})
            print(f"{label:<38} {joint}  {' '.join(parts)}")
        except Exception as e:
            print(f"{label:<38} ПОМИЛКА {type(e).__name__}: {str(e)[:70]}")

    print()
    groups: dict[str, list[str]] = {}
    for r in rows:
        groups.setdefault(r["joint"], []).append(r["target"])
    for joint, members in groups.items():
        mark = "★ ІДЕНТИЧНІ ВАГИ" if len(members) > 1 else "унікальні"
        print(f"{joint}  {mark}: {', '.join(members)}")
    print()
    print(json.dumps(rows, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
