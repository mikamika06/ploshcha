"""Крок 0 діагностики Lapa (L1): найдешевші рішучі відсіювання, усе hosted.

E1  мова промпту: Lapa англ. проти укр. — обвал глобальний чи UA-специфічний?
E2  форма задачі: абсолютна оцінка проти попарного порівняння — канал числа чи поняття?
E3  канал відповіді: слово (низька/сер/висока) проти integer — винен числовий канал?
E4  compositionality: два під-питання окремо проти синтезу — брак знань чи брак звʼязування?
E16 версія: /v1/models — що шлюз каже про модель.
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

LAPA = os.environ.get("LAPA_MODEL", "")
MAMAY = os.environ.get("MAMAY_MODEL", "")


def entropy(logprobs: list[float]) -> float:
    probs = [math.exp(lp) for lp in logprobs]
    total = sum(probs)
    if total <= 0:
        return 0.0
    return -sum((p / total) * math.log(p / total) for p in probs if p > 0)


def first_token_entropy(client, model, prompt) -> float:
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=1.0, max_tokens=6, logprobs=True, top_logprobs=20,
    )
    tok = r.choices[0].logprobs.content[0]
    return entropy([t.logprob for t in tok.top_logprobs])


def say(client, model, prompt, *, system=None, temperature=0.0, max_tokens=80) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    r = client.chat.completions.create(
        model=model, messages=msgs, temperature=temperature, max_tokens=max_tokens
    )
    return (r.choices[0].message.content or "").strip()


# ── E1: мова промпту ─────────────────────────────────────────────────────────

UK_PROMPTS = [
    "Придумай імʼя сільській дівчині.",
    "Що смачніше: борщ чи вареники? Одним словом.",
    "Назви одну річ, яку коваль тримає в кузні.",
    "Продовж: «Уранці над селом»",
]
EN_PROMPTS = [
    "Come up with a name for a village girl.",
    "Which is tastier: soup or dumplings? One word.",
    "Name one thing a blacksmith keeps in the forge.",
    "Continue: 'In the morning over the village'",
]


def e1(client):
    print(f"\n{'='*78}\nE1  мова промпту (ентропія, чим більше тим здоровіше)\n{'='*78}")
    print(f"{'модель':<12} {'укр серед.':>11} {'англ серед.':>12}   вирок")
    for model in (LAPA, MAMAY):
        if not model:
            continue
        uk = [first_token_entropy(client, model, p) for p in UK_PROMPTS]
        en = [first_token_entropy(client, model, p) for p in EN_PROMPTS]
        uk_m, en_m = sum(uk) / len(uk), sum(en) / len(en)
        verdict = ""
        if model == LAPA:
            verdict = "англ. теж обвалена -> ГЛОБАЛЬНИЙ" if en_m < 0.3 else "англ. здорова -> UA-специфічний"
        print(f"{model[:12]:<12} {uk_m:>11.3f} {en_m:>12.3f}   {verdict}")


# ── E2: абсолютна оцінка проти попарного порівняння ──────────────────────────

PAIRS = [
    ("сусід підійшов", "у селі весілля"),
    ("пішов дощ", "згоріла клуня"),
    ("принесли молоко", "померла стара сусідка"),
]


def e2(client):
    print(f"\n{'='*78}\nE2  абсолютна оцінка vs попарне порівняння\n{'='*78}")
    for model in (LAPA, MAMAY):
        if not model:
            continue
        print(f"\n  {model[:34]}")
        correct = 0
        for a, b in PAIRS:
            q = (
                f"Що важливіше для селянина: «{a}» чи «{b}»? "
                f"Відповідай рівно одним словом — перше або друге."
            )
            out = say(client, model, q).lower()
            picked_b = "друг" in out or b.split()[0] in out
            correct += picked_b  # b завжди важливіша подія
            print(f"    «{a}» vs «{b}» -> {out[:40]}")
        print(f"    правильних порівнянь: {correct}/{len(PAIRS)} (очікуємо всі 'друге')")


# ── E3: канал відповіді слово vs число ───────────────────────────────────────

E3_EVENTS = ["сусід підійшов", "у неділю весілля", "згоріла клуня в Петра"]


def e3(client):
    print(f"\n{'='*78}\nE3  канал відповіді: слово vs число\n{'='*78}")
    for model in (LAPA, MAMAY):
        if not model:
            continue
        nums, words = [], []
        for ev in E3_EVENTS:
            n = say(client, model, f"Оціни від 1 до 10 важливість для селянина: «{ev}». Лише число.")
            w = say(client, model, f"Наскільки важливо для селянина «{ev}»? Відповідай: низька, середня або висока.")
            nums.append(re.search(r"\d+", n).group() if re.search(r"\d+", n) else n[:6])
            words.append(w[:10])
        print(f"  {model[:34]}")
        print(f"    числом:  {nums}")
        print(f"    словом:  {words}")


# ── E4: compositionality-gap ─────────────────────────────────────────────────


def e4(client):
    print(f"\n{'='*78}\nE4  compositionality: знає факти окремо, але чи звʼязує?\n{'='*78}")
    facts = (
        "Факт 1: Оксана попросила Остапа підкувати коней до неділі, бо весілля.\n"
        "Факт 2: Дід Свирид, помічник Остапа, захворів і не може працювати."
    )
    for model in (LAPA, MAMAY):
        if not model:
            continue
        print(f"\n  {model[:34]}")
        q1 = say(client, model, f"{facts}\n\nПитання: що Остап має зробити до неділі?")
        q2 = say(client, model, f"{facts}\n\nПитання: чому Остапу буде важче встигнути?")
        synth = say(
            client, model,
            f"{facts}\n\nЗроби ОДИН висновок, що поєднує обидва факти. Одне речення.",
            max_tokens=100,
        )
        print(f"    під-питання 1 (знання): {q1[:70]}")
        print(f"    під-питання 2 (знання): {q2[:70]}")
        print(f"    синтез (звʼязування):   {synth[:90]}")
        links = ("бо" in synth or "тому" in synth or "оскільки" in synth) and (
            "хвор" in synth.lower() or "свирид" in synth.lower()
        )
        print(f"    -> звʼязало обидва факти: {'ТАК' if links else 'ні (переказ)'}")


# ── E16: версія моделі ───────────────────────────────────────────────────────


def e16(client):
    print(f"\n{'='*78}\nE16  що шлюз каже про модель (/v1/models)\n{'='*78}")
    try:
        models = client.models.list()
        for m in models.data:
            print(f"  id={m.id}  created={getattr(m, 'created', '—')}  owned_by={getattr(m, 'owned_by', '—')}")
    except Exception as e:
        print(f"  /v1/models недоступний: {type(e).__name__}: {e}")


def main():
    key, url = os.environ.get("LAPA_API_KEY"), os.environ.get("LAPA_BASE_URL")
    if not key:
        print("нема LAPA_API_KEY")
        return 1
    client = OpenAI(base_url=url, api_key=key, timeout=180)
    e16(client)
    e1(client)
    e2(client)
    e3(client)
    e4(client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
