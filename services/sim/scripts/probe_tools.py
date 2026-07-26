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
                "properties": {"year": {"type": "integer"}, "event": {"type": "string"}},
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

SYS_TOOLS = "Ти агент з інструментами. НЕ відповідай напряму — ОБОВʼЯЗКОВО виклич інструмент."

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

NATIVE_MODES = [
    ("auto", "auto", None),
    ("auto+sys", "auto", SYS_TOOLS),
    ("required", "required", None),
    ("required+sys", "required", SYS_TOOLS),
]


def native(client, model, prompt, tool_choice, sys):
    msgs = ([{"role": "system", "content": sys}] if sys else []) + [{"role": "user", "content": prompt}]
    try:
        r = client.chat.completions.create(
            model=model, messages=msgs, tools=TOOLS, tool_choice=tool_choice, temperature=0.0, max_tokens=200,
        )
        m = r.choices[0].message
        if m.tool_calls:
            tc = m.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = None
            return tc.function.name, args
        return None, None
    except Exception as e:
        return f"ERR:{type(e).__name__}", None


def prompt_mode(client, model, prompt):
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_B}, {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=200,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "toolcall", "schema": WIRE_SCHEMA, "strict": True}},
        )
        obj = json.loads((r.choices[0].message.content or "").strip())
        return obj.get("tool")
    except Exception as e:
        return f"ERR:{type(e).__name__}"


def run(label, model, base_url, api_key):
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=180)
    print(f"\n{'='*80}\n{label}   {model}\n{'='*80}")

    print("--- режим A: нативний tools (по режимах tool_choice) ---")
    for mode_label, tc, sys in NATIVE_MODES:
        calls = correct = 0
        for prompt, want in CASES:
            name, _ = native(client, model, prompt, tc, sys)
            calls += bool(name and not str(name).startswith("ERR"))
            correct += name == want
        print(f"  {mode_label:<14} викликів {calls}/{len(CASES)}, правильний тул {correct}/{len(CASES)}")

    print("--- режим B: prompt-based (наша схема) ---")
    json_ok = correct = 0
    for prompt, want in CASES:
        picked = prompt_mode(client, model, prompt)
        json_ok += bool(picked and not str(picked).startswith("ERR"))
        correct += picked == want
    print(f"  валідних {json_ok}/{len(CASES)}, правильний тул {correct}/{len(CASES)}")


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
