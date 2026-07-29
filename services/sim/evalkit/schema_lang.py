import re

CYRILLIC = re.compile(r"[а-яїієґА-ЯЇІЄҐ]")
LATIN = re.compile(r"[A-Za-z]")


def schema_keys(schema: dict) -> list[str]:
    keys: list[str] = []
    for variant in schema.get("oneOf", [schema]):
        keys.extend(variant.get("properties", {}))
    return keys


def latin_key_share(schema: dict) -> float:
    keys = schema_keys(schema)
    if not keys:
        return 0.0
    latin = sum(1 for k in keys if LATIN.search(k) and not CYRILLIC.search(k))
    return latin / len(keys)


def value_latin_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if c.isascii()) / len(letters) if letters else 0.0


def ukrainian_schema(fields: dict[str, str], required: list[str] | None = None) -> dict:
    """Схема з українськими ключами: латинські ключі вироджують вивід (UA-hardness §4.6)."""
    props = {name: {"type": kind} for name, kind in fields.items()}
    return {
        "type": "object",
        "properties": props,
        "required": required if required is not None else sorted(props),
        "additionalProperties": False,
    }
