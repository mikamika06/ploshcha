import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

BASE_URL = "http://127.0.0.1:8080/v1"
# Шлях рахуємо від файлу, а не від $HOME: репо не зобовʼязане лежати в ~/ploshcha.
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "research" / ".local_grid.jsonl"

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
EN_PROMPTS = [
    "Come up with a name for a village girl.",
    "Which is tastier: soup or dumplings? One word.",
    "Continue: 'In the morning over the village'",
]


def entropy(logprobs):
    ps = [math.exp(lp) for lp in logprobs]
    t = sum(ps)
    return -sum((p / t) * math.log(p / t) for p in ps if p > 0) if t > 0 else 0.0


def first_token(client, prompt):
    r = client.chat.completions.create(
        model="local", messages=[{"role": "user", "content": prompt}],
        temperature=1.0, max_tokens=6, logprobs=True, top_logprobs=20,
    )
    lps = sorted((t.logprob for t in r.choices[0].logprobs.content[0].top_logprobs), reverse=True)
    return entropy(lps), (lps[0] - lps[1] if len(lps) > 1 else 0.0)


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "local"
    client = OpenAI(base_url=BASE_URL, api_key="dummy", timeout=180)
    print(f"{'='*70}\n{label}   (llama-server :8080)\n{'='*70}")
    print(f"{'тип':<10} {'ентропія':>9} {'відрив':>8}  промпт")

    uk = []
    for kind, p in PROMPTS:
        e, m = first_token(client, p)
        uk.append((kind, e, m))
        print(f"{kind:<10} {e:>9.3f} {m:>8.2f}  {p[:44]}")
    uk_mean = sum(e for _, e, _ in uk) / len(uk)
    en = [first_token(client, p)[0] for p in EN_PROMPTS]
    en_mean = sum(en) / len(en)

    print(f"\n  СЕРЕДНЯ ентропія (укр): {uk_mean:.3f}")
    print(f"  СЕРЕДНЯ ентропія (англ): {en_mean:.3f}")
    print(f"  вирок: {'ОБВАЛЕНА' if uk_mean < 0.3 else 'здорова'}")

    row = {"label": label, "uk_mean": round(uk_mean, 4), "en_mean": round(en_mean, 4),
           "by_prompt": [{"kind": k, "ent": round(e, 4), "margin": round(m, 2)} for k, e, m in uk]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n  дописано у {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
