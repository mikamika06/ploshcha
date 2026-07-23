"""Абляція: ЯКА САМЕ фіча JSON Schema ламає constrained decoding на бекенді.

Метод: адверсарний промпт (просимо прозу). Три можливі вироки:
  ENFORCED  — вивід валідний за схемою -> граматика скомпілювалась
  JSON_ONLY — валідний JSON, але схемі не відповідає -> бекенд відкотився в json-режим
  IGNORED   — не JSON (проза) -> параметр повністю проігноровано
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

ADV = "Розкажи довгу казку про діда та ріпку. Пиши прозою, щонайменше пʼять речень."

STR = {"type": "string"}

CASES: list[tuple[str, dict]] = [
    # ── база, що вже працює ──
    ("flat + enum", {
        "type": "object",
        "properties": {"type": {"type": "string", "enum": ["wait", "speak"]}, "text": STR},
        "required": ["type"], "additionalProperties": False}),
    # ── по одній фічі зверху ──
    ("flat + const", {
        "type": "object",
        "properties": {"type": {"const": "wait"}, "text": STR},
        "required": ["type"], "additionalProperties": False}),
    ("flat + additionalProperties:true", {
        "type": "object",
        "properties": {"type": {"type": "string", "enum": ["wait", "speak"]}},
        "required": ["type"], "additionalProperties": True}),
    ("flat + без required", {
        "type": "object",
        "properties": {"type": {"type": "string", "enum": ["wait", "speak"]}},
        "additionalProperties": False}),
    ("nested object", {
        "type": "object",
        "properties": {"type": {"type": "string", "enum": ["wait"]},
                       "payload": {"type": "object", "properties": {"reason": STR},
                                   "required": ["reason"], "additionalProperties": False}},
        "required": ["type", "payload"], "additionalProperties": False}),
    ("$defs + $ref", {
        "$defs": {"Reason": STR},
        "type": "object",
        "properties": {"type": {"type": "string", "enum": ["wait"]},
                       "reason": {"$ref": "#/$defs/Reason"}},
        "required": ["type", "reason"], "additionalProperties": False}),
    ("oneOf (інлайн, без $ref)", {
        "oneOf": [
            {"type": "object", "properties": {"type": {"const": "wait"}, "reason": STR},
             "required": ["type", "reason"], "additionalProperties": False},
            {"type": "object", "properties": {"type": {"const": "speak"}, "text": STR},
             "required": ["type", "text"], "additionalProperties": False},
        ]}),
    ("anyOf (інлайн, без $ref)", {
        "anyOf": [
            {"type": "object", "properties": {"type": {"const": "wait"}},
             "required": ["type"], "additionalProperties": False},
            {"type": "object", "properties": {"type": {"const": "speak"}, "text": STR},
             "required": ["type", "text"], "additionalProperties": False},
        ]}),
    ("oneOf + $ref (без discriminator)", {
        "$defs": {
            "Wait": {"type": "object", "properties": {"type": {"const": "wait"}},
                     "required": ["type"], "additionalProperties": False},
            "Speak": {"type": "object", "properties": {"type": {"const": "speak"}, "text": STR},
                      "required": ["type", "text"], "additionalProperties": False}},
        "oneOf": [{"$ref": "#/$defs/Wait"}, {"$ref": "#/$defs/Speak"}]}),
    ("НАША union (pydantic, з discriminator)", action_json_schema()),
]


def verdict(text: str, schema: dict) -> str:
    try:
        obj = json.loads(text)
    except Exception:
        return "IGNORED"
    try:
        import jsonschema

        jsonschema.validate(obj, schema)
        return "ENFORCED"
    except Exception:
        return "JSON_ONLY"


def run(label: str, model: str, base_url: str, api_key: str) -> None:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=180)
    print(f"\n{'=' * 72}\n{label}   {model}\n{'=' * 72}")
    print(f"{'схема':<40} {'вирок':<11} вивід")
    for name, schema in CASES:
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": ADV}],
                temperature=0.0,
                max_tokens=400,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "probe", "schema": schema, "strict": True},
                },
            )
            text = (r.choices[0].message.content or "").strip()
            print(f"{name:<40} {verdict(text, schema):<11} {text[:70].replace(chr(10), ' ')}")
        except Exception as e:
            print(f"{name:<40} {'ERROR':<11} {type(e).__name__}: {str(e)[:60]}")


def main() -> int:
    if os.environ.get("LAPA_API_KEY"):
        run("ХОСТОВАНИЙ Lapathoniia", os.environ["LAPA_MODEL"],
            os.environ["LAPA_BASE_URL"], os.environ["LAPA_API_KEY"])
    if os.environ.get("LOCAL_BASE_URL"):
        run("ЛОКАЛЬНИЙ", os.environ.get("LOCAL_MODEL", "local"),
            os.environ["LOCAL_BASE_URL"], os.environ.get("LOCAL_API_KEY", "EMPTY"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
