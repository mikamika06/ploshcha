import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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

from ploshcha_sim.adapters import FakeToolbox
from ploshcha_sim.adapters.llm_openai import OpenAICompatLlm

SHORT = "Опиши одним реченням, як виглядає українське село восьмого березня."

LONG_PREFIX = (
    "Ти агент з інструментами: check_date(year,event), lookup_fact(entity), calc(expr), final_answer(text). "
    "Аргументи пиши УКРАЇНСЬКОЮ дослівно як у задачі. Перевіряй факти інструментом."
)
LONG_TASK = (
    "Задача: Перевір, чи Битва під Крутами відбулася 1918 року, і обчисли, скільки років минуло "
    "від початку Хмельниччини до неї.\n"
    "Виклик: {\"tool\": \"check_date\", \"year\": 1918, \"event\": \"Битва під Крутами\"}\n"
    "Результат: {\"matches\": true, \"actual_year\": 1918, \"known\": true}\n"
    "Виклик: {\"tool\": \"lookup_fact\", \"entity\": \"Битва під Крутами\"}\n"
    "Результат: {\"fact\": \"Бій 29 січня 1918 року між військами УНР і більшовиками.\", \"known\": true}\n"
    "Наступний крок — один JSON (виклик інструмента або final_answer):"
)
N = 5


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def report(label, outs):
    uniq = len(set(outs))
    mark = "СТАБІЛЬНО" if uniq == 1 else f"РОЗКИД {uniq}"
    print(f"   {label:<34} {uniq}/{len(outs)} унікальних  [{mark}]  {Counter(outs).most_common(3)}")
    return uniq


def short_matrix(llm):
    def call(temperature, seed):
        return h(llm.generate(SHORT, temperature=temperature, max_tokens=80, seed=seed).text)

    print("  короткий промпт, без схеми:")
    report("A t=0 однаковий seed", [call(0.0, 1) for _ in range(N)])
    report("B t=0 різні seed", [call(0.0, s) for s in range(1, N + 1)])
    report("C t=0 seed=None", [call(0.0, None) for _ in range(N)])
    report("D t=0.8 однаковий seed", [call(0.8, 1) for _ in range(N)])
    report("E t=0.8 різні seed", [call(0.8, s) for s in range(1, N + 1)])


def realistic(llm, schema):
    def call(seed=1):
        return h(llm.generate_structured(LONG_TASK, schema, system=LONG_PREFIX,
                                         temperature=0.0, max_tokens=256, seed=seed).text)

    print("  довгий промпт + json_schema (як в оркестраторі), t=0:")
    report("F послідовно, той самий seed", [call() for _ in range(N)])
    with ThreadPoolExecutor(max_workers=N) as pool:
        parallel = list(pool.map(lambda _: call(), range(N)))
    report("G паралельно, той самий seed", parallel)
    report("H послідовно, різні seed", [call(s) for s in range(1, N + 1)])


def main():
    key, url = os.environ.get("LAPA_API_KEY"), os.environ.get("LAPA_BASE_URL")
    if not key:
        print("нема LAPA_API_KEY")
        return 1
    schema = FakeToolbox().wire_schema()
    for name in ("MAMAY_MODEL", "LAPA_MODEL"):
        model = os.environ[name]
        llm = OpenAICompatLlm(model=model, base_url=url, api_key=key, structured_mode="json_schema")
        print("=" * 78)
        print(f"МОДЕЛЬ {name}={model}")
        short_matrix(llm)
        realistic(llm, schema)
    print("\nСхема виводу: 1/N унікальних = бітово стабільно; >1 = недетермінізм бекенда.")
    print(json.dumps({"endpoint": url, "n": N}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
