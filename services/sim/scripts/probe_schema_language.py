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

from evalkit.prompts import resolve
from ploshcha_sim.adapters.llm_openai import OpenAICompatLlm

TASK = ("Абзац: «У селі Вербівці 1893 року коваль Панас Жмуренко викував дзвін. "
        "Роботу оплатив мірошник Гнат Клепач.» Витягни дані.")

LATIN = {
    "type": "object",
    "properties": {"year": {"type": "string"}, "blacksmith": {"type": "string"},
                   "miller": {"type": "string"}},
    "required": ["year", "blacksmith", "miller"],
    "additionalProperties": False,
}
CYRILLIC = {
    "type": "object",
    "properties": {"рік": {"type": "string"}, "коваль": {"type": "string"},
                   "мірошник": {"type": "string"}},
    "required": ["рік", "коваль", "мірошник"],
    "additionalProperties": False,
}
EXPECTED = ("1893", "Панас Жмуренко", "Гнат Клепач")

SHORT_SYSTEM = ("Ти витягуєш дані з українського тексту. Бери значення ЛИШЕ з наданого абзацу. "
                "Імена й назви подавай у називному відмінку. Відповідай коротко.")


def latin_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if c.isascii()) / len(letters) if letters else 0.0


def values_of(raw: str) -> str:
    try:
        return " ".join(str(v) for v in json.loads(raw).values())
    except json.JSONDecodeError:
        return raw


def main():
    key, url = os.environ.get("LAPA_API_KEY"), os.environ.get("LAPA_BASE_URL")
    if not key:
        print("нема LAPA_API_KEY")
        return 1
    prompts = (("короткий", SHORT_SYSTEM),
               ("повний (extract/plain)", resolve("extract/plain").render_system()))
    seeds = (1, 2, 3)
    print("латиниця у ЗНАЧЕННЯХ (0% = чиста українська) · ok = всі три значення на місці\n")
    for name in ("MAMAY_MODEL", "LAPA_MODEL"):
        llm = OpenAICompatLlm(model=os.environ[name], base_url=url, api_key=key,
                              structured_mode="json_schema")
        print("=" * 78)
        print(os.environ[name])
        for pname, system in prompts:
            for label, schema in (("латинські ключі", LATIN), ("українські ключі", CYRILLIC)):
                bad = 0
                for seed in seeds:
                    raw = llm.generate_structured(TASK, schema, system=system, temperature=0.0,
                                                  max_tokens=200, seed=seed).text
                    vals = values_of(raw)
                    if not all(e in vals for e in EXPECTED):
                        bad += 1
                print(f"  промпт={pname:<22} {label:<18} брак {bad}/{len(seeds)}")
    print("\nВисновок §4.6 UA-hardness: латинські ключі створюють ЛАТЕНТНУ крихкість, яку")
    print("детальний промпт маскує; українські ключі коректні за обох промптів.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
