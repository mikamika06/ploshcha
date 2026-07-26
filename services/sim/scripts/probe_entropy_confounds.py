"""Три метрологічні конфаунди, що можуть убити результат «ентропія Lapa 19× менша».

Кожен — дешева hosted-перевірка. Якщо хоч один спрацює, заголовок L0 недійсний.

C1 (temp-scaled logprobs): чи шлюз віддає СИРІ логіти, чи вже поділені на serving-temperature?
   Якщо logprobs залежать від параметра temperature у запиті — вони пост-температурні, і
   порівняння Lapa vs Mamay = порівняння конфігів деплою, не ваг.
C2 (digit tokenization): Gemma ріже числа поцифрово, «10» = «1»+«0». Ентропія ПЕРШОГО токена
   на шкалі 1..10 плутає значення 1 із префіксом 10. Рахуємо повний P(value) по 1..10.
C3 (single-item conditioning): обвал міряний на пакеті=1. На пакеті=6 Lapa показувала розкид.
   Рахуємо ентропію per-item при різних розмірах пакета.
"""

import json
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

RATE_PROMPT = "Оціни від 1 до 10, наскільки для селянина важливо, що в неділю весілля в Ганни. Лише число."


def top_logprobs_first(client, model, prompt, temperature):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=6,
        logprobs=True,
        top_logprobs=20,
    )
    tok = r.choices[0].logprobs.content[0]
    return sorted((t.logprob for t in tok.top_logprobs), reverse=True)


def entropy(logprobs):
    probs = [math.exp(lp) for lp in logprobs]
    total = sum(probs)
    if total <= 0:
        return 0.0
    return -sum((p / total) * math.log(p / total) for p in probs if p > 0)


# ── C1: температурне масштабування logprobs ──────────────────────────────────


def confound_temp_scaling(client, model):
    print(f"\n--- C1 temp-scaled logprobs? --- {model}")
    rows = []
    for temp in (0.01, 1.0, 2.0):
        lps = top_logprobs_first(client, model, RATE_PROMPT, temp)
        margin = lps[0] - lps[1]
        rows.append((temp, entropy(lps), margin))
        print(f"  temperature={temp:<5} ентропія(top20)={entropy(lps):.4f}  відрив top1-2={margin:.2f}")
    ents = [r[1] for r in rows]
    verdict = "СИРІ (ентропія стабільна)" if max(ents) - min(ents) < 0.05 else "ПОСТ-ТЕМПЕРАТУРНІ — 19× НЕДІЙСНИЙ"
    print(f"  вирок: {verdict}")


# ── C2: повний розподіл по значеннях 1..10 через constrained-схему ───────────


VALUE_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "integer"}},
    "required": ["value"],
    "additionalProperties": False,
}


def confound_digit_tokenization(client, model):
    """Просимо {"value": N}. Дивимось logprobs на позиції, де стоїть цифра.

    Повне P(1..10) не дістати без forced-decoding кожного значення (дорого), тож
    робимо дешевший, але коректний прокладень: беремо logprobs на позиції першої
    цифри значення й ОКРЕМО дивимось, чи «1» тут = значення 1 чи початок «10».
    """
    print(f"\n--- C2 digit tokenization --- {model}")
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": RATE_PROMPT}],
        temperature=1.0,
        max_tokens=8,
        logprobs=True,
        top_logprobs=20,
        response_format={"type": "json_schema", "json_schema": {"name": "v", "schema": VALUE_SCHEMA, "strict": True}},
    )
    content = r.choices[0].logprobs.content
    # знаходимо перший токен, що містить цифру
    digit_pos = next((i for i, t in enumerate(content) if re.search(r"\d", t.token)), None)
    if digit_pos is None:
        print("  цифру не знайдено в токенах")
        return
    tok = content[digit_pos]
    dist = sorted(((t.token, t.logprob) for t in tok.top_logprobs), key=lambda x: -x[1])
    digits = [(t, lp) for t, lp in dist if re.fullmatch(r"\d+", t.strip())]
    ent_digits = entropy([lp for _, lp in digits])
    print(f"  позиція цифри={digit_pos}, вивід={(r.choices[0].message.content or '').strip()[:20]}")
    print(f"  розподіл по цифрових токенах (top): {[(t, round(lp,2)) for t,lp in digits[:8]]}")
    print(f"  ентропія лише по цифрах: {ent_digits:.4f}")
    print("  (якщо '1' домінує І поряд є '10' — значення 10 недооцінене; дивись, чи є '10' у списку)")


# ── C3: залежність від розміру пакета ────────────────────────────────────────

EVENTS = [
    "Свирид підійшов.",
    "У неділю весілля в Ганни.",
    "Пішов дощ.",
    "Згоріла клуня в Петра.",
    "Помер старий Панас.",
    "Оксана принесла молоко.",
]


def rate_batch(client, model, events):
    listed = "\n".join(f"{i+1}. {e}" for i, e in enumerate(events))
    schema = {
        "type": "object",
        "properties": {"ratings": {"type": "array", "items": {"type": "integer"}}},
        "required": ["ratings"],
        "additionalProperties": False,
    }
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Оцінюй важливість подій для селянина від 1 до 10."},
            {"role": "user", "content": f"Оціни кожну:\n{listed}"},
        ],
        temperature=0.0,
        max_tokens=120,
        response_format={"type": "json_schema", "json_schema": {"name": "r", "schema": schema, "strict": True}},
    )
    try:
        return [int(v) for v in json.loads(r.choices[0].message.content)["ratings"]]
    except Exception:
        return None


def confound_batch_size(client, model):
    print(f"\n--- C3 batch-size conditioning --- {model}")
    for n in (1, 2, 4, 6):
        got = rate_batch(client, model, EVENTS[:n])
        if got is None:
            print(f"  пакет={n}: провал парсингу")
            continue
        spread = (max(got) - min(got)) if len(got) > 1 else 0
        print(f"  пакет={n}: {got}  розкид={spread}")


def run(client, model):
    print(f"\n{'='*80}\n{model}\n{'='*80}")
    confound_temp_scaling(client, model)
    confound_digit_tokenization(client, model)
    confound_batch_size(client, model)


def main():
    key, url = os.environ.get("LAPA_API_KEY"), os.environ.get("LAPA_BASE_URL")
    if not key:
        print("нема LAPA_API_KEY")
        return 1
    client = OpenAI(base_url=url, api_key=key, timeout=180)
    for var in ("LAPA_MODEL", "MAMAY_MODEL"):
        if os.environ.get(var):
            run(client, os.environ[var])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
