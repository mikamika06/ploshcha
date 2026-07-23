"""Строга перевірка constrained decoding на будь-якому OpenAI-сумісному ендпоїнті.

Ідея: адверсарний промпт (просимо ПРОЗУ). Якщо граматика справді діє — вивід усе одно
мусить бути JSON за схемою. Якщо повернулась казка — механізм проігноровано.
"""

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

from ploshcha_sim.domain import action_json_schema  # noqa: E402

ADVERSARIAL = "Розкажи довгу казку про діда та ріпку. Пиши прозою, щонайменше пʼять речень."


def mechanisms(schema: dict) -> list[tuple[str, dict]]:
    """(назва, kwargs для chat.completions.create)."""
    return [
        ("baseline (нічого)", {}),
        ("extra_body.guided_json", {"extra_body": {"guided_json": schema}}),
        (
            "extra_body.guided_json+backend",
            {"extra_body": {"guided_json": schema, "guided_decoding_backend": "xgrammar"}},
        ),
        ("extra_body.guided_choice", {"extra_body": {"guided_choice": ["move_to", "speak", "wait"]}}),
        ("extra_body.guided_regex", {"extra_body": {"guided_regex": r"\{\"type\": \"wait\"\}"}}),
        (
            "extra_body.structured_outputs",
            {"extra_body": {"structured_outputs": {"json": schema}}},
        ),
        ("response_format=json_object", {"response_format": {"type": "json_object"}}),
        (
            "response_format=json_schema",
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "action", "schema": schema, "strict": True},
                }
            },
        ),
    ]


def probe(label: str, model: str, base_url: str, api_key: str) -> None:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=120)
    schema = action_json_schema()

    print(f"\n{'=' * 70}\n{label}  model={model}\n{base_url}\n{'=' * 70}")

    try:
        served = [m.id for m in client.models.list().data]
        print(f"GET /models -> {served}")
    except Exception as e:
        print(f"GET /models -> {type(e).__name__}: {str(e)[:120]}")

    for name, kwargs in mechanisms(schema):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": ADVERSARIAL}],
                temperature=0.0,
                max_tokens=600,
                **kwargs,
            )
            text = (resp.choices[0].message.content or "").strip()
            finish = resp.choices[0].finish_reason
            is_json, matches = False, False
            try:
                obj = json.loads(text)
                is_json = True
                from ploshcha_sim.domain import parse_action

                parse_action(obj)
                matches = True
            except Exception:
                pass
            verdict = (
                "СХЕМА ДІЄ" if matches else ("JSON-режим" if is_json else "проігноровано (проза)")
            )
            print(f"\n  {name:<34} {verdict}   finish={finish}")
            print(f"    {text[:160].replace(chr(10), ' ')}")
        except Exception as e:
            msg = str(e).replace("\n", " ")[:200]
            print(f"\n  {name:<34} ПОМИЛКА {type(e).__name__}")
            print(f"    {msg}")


def main() -> int:
    targets = []
    if os.environ.get("LAPA_API_KEY"):
        targets.append(
            (
                "ХОСТОВАНИЙ Lapathoniia",
                os.environ["LAPA_MODEL"],
                os.environ["LAPA_BASE_URL"],
                os.environ["LAPA_API_KEY"],
            )
        )
    local = os.environ.get("LOCAL_BASE_URL")
    if local:
        targets.append(
            ("ЛОКАЛЬНИЙ", os.environ.get("LOCAL_MODEL", "local"), local, os.environ.get("LOCAL_API_KEY", "EMPTY"))
        )

    if not targets:
        print("нема цілей: заповни LAPA_API_KEY або LOCAL_BASE_URL")
        return 1
    for t in targets:
        probe(*t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
