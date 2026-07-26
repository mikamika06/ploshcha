"""Дія має бути строго за схемою."""

import pytest
from pydantic import ValidationError

from ploshcha_sim.domain import (
    ACTION_TYPES,
    MoveTo,
    Speak,
    Wait,
    action_json_schema,
    parse_action,
)


def test_parse_each_action_type():
    cases = [
        {"type": "move_to", "poi": "kuznya"},
        {"type": "speak", "to": ["mati"], "text": "Добридень"},
        {"type": "use_object", "poi": "krynytsia"},
        {"type": "post_to_board", "topic": "Завтра толока"},
        {"type": "reflect"},
        {"type": "wait", "reason": "нема кого питати"},
    ]
    parsed = [parse_action(c) for c in cases]
    assert [p.type for p in parsed] == list(ACTION_TYPES)


def test_discriminator_picks_right_class():
    a = parse_action({"type": "move_to", "poi": "ploshcha"})
    assert isinstance(a, MoveTo) and a.poi == "ploshcha"
    b = parse_action({"type": "speak", "text": "гей"})
    assert isinstance(b, Speak) and b.to == []


def test_speak_defaults_to_broadcast():
    assert Speak(text="усім").to == []


def test_wait_reason_optional():
    assert Wait().reason is None


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "move_to"},                      # немає обовʼязкового poi
        {"type": "speak", "text": ""},            # порожній текст
        {"type": "post_to_board", "topic": ""},   # порожня тема
        {"type": "teleport", "poi": "x"},         # невідомий тип дії
        {"poi": "kuznya"},                        # немає дискримінатора
        "move_to kuznya",                         # не обʼєкт
    ],
)
def test_invalid_actions_rejected(bad):
    with pytest.raises(ValidationError):
        parse_action(bad)


def test_json_schema_exposes_all_variants():
    schema = action_json_schema()
    blob = str(schema)
    for t in ACTION_TYPES:
        assert t in blob, f"{t} відсутній у JSON-схемі"


# ── wire-схема для constrained decoding ──────────────────────────────────────


def test_wire_schema_is_flat():
    """GBNF-бекенди не компілюють anyOf/$defs — wire-схема мусить бути пласкою."""
    from ploshcha_sim.domain import wire_action_json_schema

    blob = str(wire_action_json_schema())
    assert "anyOf" not in blob and "$defs" not in blob and "$ref" not in blob


def test_wire_schema_type_enum_matches_action_types():
    from ploshcha_sim.domain import wire_action_json_schema

    s = wire_action_json_schema()
    assert s["properties"]["type"]["enum"] == list(ACTION_TYPES)
    assert s["required"] == ["type"] and s["additionalProperties"] is False


def test_wire_schema_covers_every_variant_field():
    from ploshcha_sim.domain import wire_action_json_schema

    props = wire_action_json_schema()["properties"]
    for field in ("poi", "to", "text", "topic", "reason"):
        assert field in props, f"{field} відсутнє у wire-схемі"


def test_wire_schema_optional_field_has_plain_type():
    """Wait.reason — це str|None; у wire-схемі мусить лишитись просто string."""
    from ploshcha_sim.domain import wire_action_json_schema

    assert wire_action_json_schema()["properties"]["reason"]["type"] == "string"


def test_union_still_parses_flat_payload_with_extra_keys():
    """Модель під wire-схемою може додати поле, чуже цій дії — union його ігнорує."""
    a = parse_action({"type": "wait", "reason": "нема кого", "poi": "kuznya"})
    assert a.type == "wait" and a.reason == "нема кого"


# ── ступінь 2: посилений union ───────────────────────────────────────────────


def test_plain_union_leaves_discriminator_optional():
    """Корінь бага: pydantic не кладе поле з default у required."""
    from ploshcha_sim.domain import action_json_schema

    for name, variant in action_json_schema()["$defs"].items():
        assert "type" in variant["properties"]
        assert "type" not in variant.get("required", []), name


def test_strict_schema_requires_discriminator_everywhere():
    from ploshcha_sim.domain import strict_action_json_schema

    for name, variant in strict_action_json_schema()["$defs"].items():
        assert "type" in variant["required"], name
        assert variant["additionalProperties"] is False, name


def test_strict_schema_still_validates_real_actions():
    import jsonschema

    from ploshcha_sim.domain import strict_action_json_schema

    s = strict_action_json_schema()
    jsonschema.validate({"type": "speak", "to": ["mati"], "text": "гей"}, s)
    jsonschema.validate({"type": "wait", "reason": None}, s)


def test_strict_schema_rejects_missing_type():
    import jsonschema
    import pytest

    from ploshcha_sim.domain import strict_action_json_schema

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"text": "казка"}, strict_action_json_schema())
