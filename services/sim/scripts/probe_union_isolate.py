"""Раунд 2: що САМЕ в нашому pydantic-union ламає компіляцію граматики.

Беремо нашу union-схему і по черзі знімаємо по одній підозрі.
"""

import copy
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


def strip_keys(node, keys: set[str]):
    if isinstance(node, dict):
        return {k: strip_keys(v, keys) for k, v in node.items() if k not in keys}
    if isinstance(node, list):
        return [strip_keys(v, keys) for v in node]
    return node


def const_to_enum(node):
    """{"const": "x"} -> {"type":"string","enum":["x"]}"""
    if isinstance(node, dict):
        if "const" in node and isinstance(node["const"], str):
            return {"type": "string", "enum": [node["const"]]}
        return {k: const_to_enum(v) for k, v in node.items()}
    if isinstance(node, list):
        return [const_to_enum(v) for v in node]
    return node


BASE = action_json_schema()


def variants() -> list[tuple[str, dict]]:
    return [
        ("union як є (контроль)", copy.deepcopy(BASE)),
        ("− discriminator", strip_keys(BASE, {"discriminator"})),
        ("− default", strip_keys(BASE, {"default"})),
        ("− title", strip_keys(BASE, {"title"})),
        ("− description", strip_keys(BASE, {"description"})),
        ("− minLength", strip_keys(BASE, {"minLength"})),
        ("const -> enum", const_to_enum(copy.deepcopy(BASE))),
        ("− discriminator − default", strip_keys(BASE, {"discriminator", "default"})),
        ("− discriminator, const->enum", const_to_enum(strip_keys(BASE, {"discriminator"}))),
        (
            "усе прибрано",
            const_to_enum(
                strip_keys(BASE, {"discriminator", "default", "title", "description", "minLength"})
            ),
        ),
    ]


def verdict(text: str, finish: str, schema: dict) -> str:
    if finish == "length":
        return "TRUNC"
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
    print(f"\n{'=' * 74}\n{label}   {model}\n{'=' * 74}")
    print(f"{'варіант схеми':<32} {'вирок':<10} вивід")
    for name, schema in variants():
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": ADV}],
                temperature=0.0,
                max_tokens=250,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "probe", "schema": schema, "strict": True},
                },
            )
            text = (r.choices[0].message.content or "").strip()
            fin = r.choices[0].finish_reason
            print(f"{name:<32} {verdict(text, fin, schema):<10} {text[:64].replace(chr(10), ' ')}")
        except Exception as e:
            print(f"{name:<32} {'ERROR':<10} {type(e).__name__}: {str(e)[:55]}")


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
