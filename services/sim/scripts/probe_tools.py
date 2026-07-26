import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load_env(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


load_env(ROOT / ".env")

from openai import OpenAI

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_date",
            "description": "Перевірити, чи рік відповідає історичній події.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer"},
                    "event": {"type": "string"},
                },
                "required": ["year", "event"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_fact",
            "description": "Знайти факт про сутність (людину, місце, подію).",
            "parameters": {
                "type": "object",
                "properties": {"entity": {"type": "string"}},
                "required": ["entity"],
                "additionalProperties": False,
            },
        },
    },
]

CASES = [
    ("У якому році була Битва під Крутами? Скористайся інструментом.", "lookup_fact"),
    ("Перевір, чи 1648 рік відповідає початку Хмельниччини. Виклич інструмент.", "check_date"),
    ("Знайди факт про Тараса Шевченка.", "lookup_fact"),
]

WIRE_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": ["check_date", "lookup_fact"]},
        "year": {"type": "integer"},
        "event": {"type": "string"},
        "entity": {"type": "string"},
    },
    "required": ["tool"],
    "additionalProperties": False,
}
SYSTEM_B = (
    "Ти обираєш ОДИН інструмент і повертаєш РІВНО один JSON: "
    '{"tool":"<check_date|lookup_fact>", ...аргументи}. Жодного тексту поза JSON. '
    "check_date(year,event); lookup_fact(entity)."
)


def try_native(client, model, prompt):
    try:
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            tools=TOOLS, tool_choice="auto", temperature=0.0, max_tokens=200,
        )
        msg = r.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = None
            return "NATIVE-OK", {"tool": tc.function.name, "args": args}, ""
        return "NO-TOOLCALL", None, (msg.content or "")[:60]
    except Exception as e:
        return f"ERR:{type(e).__name__}", None, str(e)[:70]


def try_prompt(client, model, prompt):
    t = ""
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_B}, {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=200,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "toolcall", "schema": WIRE_SCHEMA, "strict": True}},
        )
        t = (r.choices[0].message.content or "").strip()
        obj = json.loads(t)
        return "JSON-OK", {"tool": obj.get("tool"), "args": {k: v for k, v in obj.items() if k != "tool"}}, t[:60]
    except json.JSONDecodeError:
        return "NOT-JSON", None, t[:60]
    except Exception as e:
        return f"ERR:{type(e).__name__}", None, str(e)[:70]


def run(label, model, base_url, api_key):
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=180)
    print(f"\n{'='*80}\n{label}   {model}\n{'='*80}")

    print("--- режим A: нативний tools ---")
    native_ok = sel_ok = 0
    for prompt, want_tool in CASES:
        verdict, call, raw = try_native(client, model, prompt)
        picked = call["tool"] if call else "—"
        native_ok += verdict == "NATIVE-OK"
        sel_ok += bool(call and call["tool"] == want_tool)
        print(f"  {verdict:<12} тул={picked:<12} (треба {want_tool})  {raw}")
    print(f"  нативних викликів: {native_ok}/{len(CASES)}, правильний тул: {sel_ok}/{len(CASES)}")

    print("--- режим B: prompt-based ---")
    json_ok = selb_ok = 0
    for prompt, want_tool in CASES:
        verdict, call, raw = try_prompt(client, model, prompt)
        picked = call["tool"] if call else "—"
        json_ok += verdict == "JSON-OK"
        selb_ok += bool(call and call["tool"] == want_tool)
        print(f"  {verdict:<12} тул={picked:<12} (треба {want_tool})  {raw}")
    print(f"  валідних JSON: {json_ok}/{len(CASES)}, правильний тул: {selb_ok}/{len(CASES)}")


def main():
    key, url = os.environ.get("LAPA_API_KEY"), os.environ.get("LAPA_BASE_URL")
    if not key:
        print("нема LAPA_API_KEY")
        return 1
    for var in ("LAPA_MODEL", "MAMAY_MODEL"):
        if os.environ.get(var):
            run(var, os.environ[var], url, key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
