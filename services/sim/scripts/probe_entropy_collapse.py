"""Кількісно: чи справді розподіл Lapa загострений, і чи шлюз кешує відповіді.

Дві незалежні речі:
  A. Ентропія розподілу першого згенерованого токена по набору різних промптів.
     Загострений (near-one-hot) розподіл = мала ентропія + великий відрив top1 від top2.
     Це системна властивість моделі, а не властивість задачі оцінювання.
  B. Чи дає шлюз справжній семплінг: той самий промпт кілька разів, різні seed,
     промпт із нонсом. Якщо все однакове — pass^k та будь-яка метрика на повторах
     на цьому API неможлива.
"""

import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


load_env(ROOT / ".env")

from openai import OpenAI  # noqa: E402

# Промпти навмисно різнорідні: суб'єктивна оцінка, вибір, продовження, факт.
# На суб'єктивних і відкритих модель МУСИТЬ вагатись — якщо не вагається ніде,
# це властивість моделі, не задачі.
PROMPTS = [
    ("оцінка", "Оціни від 1 до 10, наскільки важливо для селянина, що сусід позичив сокиру. Лише число."),
    ("оцінка", "Оціни від 1 до 10, наскільки важливо для селянина, що в селі весілля. Лише число."),
    ("оцінка", "Оціни від 1 до 10, наскільки важливо для селянина, що пішов дощ. Лише число."),
    ("вибір", "Коваль чи мірошник більше знає про залізо? Відповідай одним словом."),
    ("вибір", "Що селянин зробить уранці першим: подоїть корову чи піде на площу? Одним словом."),
    ("відкрите", "Придумай імʼя сільській дівчині."),
    ("відкрите", "Назви одну річ, яку коваль тримає в кузні."),
    ("відкрите", "Продовж: «Уранці над селом»"),
    ("відкрите", "Яка погода найкраща для толоки?"),
    ("факт", "Столиця України — це"),
    ("факт", "Два плюс два дорівнює"),
    ("смак", "Що смачніше: борщ чи вареники? Одним словом."),
]


def first_token_dist(client: OpenAI, model: str, prompt: str) -> list[float] | None:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_tokens=8,
        logprobs=True,
        top_logprobs=20,
    )
    try:
        tok = r.choices[0].logprobs.content[0]
        return [t.logprob for t in tok.top_logprobs]
    except Exception:
        return None


def entropy_of(logprobs: list[float]) -> float:
    """Ентропія по top-k, з перенормуванням (нижня оцінка справжньої)."""
    probs = [math.exp(lp) for lp in logprobs]
    total = sum(probs)
    if total <= 0:
        return 0.0
    return -sum((p / total) * math.log(p / total) for p in probs if p > 0)


def run_entropy(client: OpenAI, model: str) -> None:
    print(f"\n{'=' * 78}\nА. ЗАГОСТРЕНІСТЬ РОЗПОДІЛУ — {model}\n{'=' * 78}")
    print(f"{'тип':<10} {'ентропія':>9} {'відрив':>9}  промпт")
    rows: list[tuple[str, float, float]] = []
    for kind, prompt in PROMPTS:
        lps = first_token_dist(client, model, prompt)
        if lps is None:
            print("  logprobs недоступні")
            return
        ent = entropy_of(lps)
        # top_logprobs приходять НЕ завжди відсортованими (перевірено на Mamay),
        # тож відрив рахуємо по відсортованих, інакше виходять від'ємні значення
        ranked = sorted(lps, reverse=True)
        margin = (ranked[0] - ranked[1]) if len(ranked) > 1 else 0.0
        rows.append((kind, ent, margin))
        print(f"{kind:<10} {ent:>9.3f} {margin:>9.2f}  {prompt[:44]}")

    by_kind: dict[str, list[tuple[float, float]]] = {}
    for kind, ent, margin in rows:
        by_kind.setdefault(kind, []).append((ent, margin))
    print("\n  середнє за типом:")
    for kind, vals in by_kind.items():
        e = sum(v[0] for v in vals) / len(vals)
        m = sum(v[1] for v in vals) / len(vals)
        print(f"    {kind:<10} ентропія {e:.3f}   відрив {m:.2f} нат")
    all_e = sum(r[1] for r in rows) / len(rows)
    all_m = sum(r[2] for r in rows) / len(rows)
    print(f"    {'УСЬОГО':<10} ентропія {all_e:.3f}   відрив {all_m:.2f} нат")


def run_sampling(client: OpenAI, model: str) -> None:
    print(f"\n{'=' * 78}\nБ. ЧИ ДАЄ ШЛЮЗ СПРАВЖНІЙ СЕМПЛІНГ — {model}\n{'=' * 78}")
    base = "Придумай імʼя сільській дівчині. Лише імʼя."

    def gen(**extra) -> str:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": extra.pop("prompt", base)}],
            temperature=1.0,
            max_tokens=16,
            **extra,
        )
        return (r.choices[0].message.content or "").strip()[:40]

    same = [gen() for _ in range(4)]
    print(f"  той самий промпт ×4         унікальних {len(set(same))}/4  {same}")

    seeded = []
    for s in (1, 2, 3, 4):
        try:
            seeded.append(gen(seed=s))
        except Exception as e:
            seeded.append(f"ERR:{type(e).__name__}")
    print(f"  різні seed 1-4              унікальних {len(set(seeded))}/4  {seeded}")

    nonced = [gen(prompt=f"{base} (варіант {i})") for i in range(1, 5)]
    print(f"  промпт із нонсом            унікальних {len(set(nonced))}/4  {nonced}")


def main() -> int:
    key, url = os.environ.get("LAPA_API_KEY"), os.environ.get("LAPA_BASE_URL")
    if not key:
        print("нема LAPA_API_KEY")
        return 1
    client = OpenAI(base_url=url, api_key=key, timeout=180)
    for var in ("LAPA_MODEL", "MAMAY_MODEL"):
        if os.environ.get(var):
            run_entropy(client, os.environ[var])
    for var in ("LAPA_MODEL", "MAMAY_MODEL"):
        if os.environ.get(var):
            run_sampling(client, os.environ[var])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
