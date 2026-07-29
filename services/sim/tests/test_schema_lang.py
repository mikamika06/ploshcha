from evalkit.schema_lang import (
    latin_key_share,
    schema_keys,
    ukrainian_schema,
    value_latin_share,
)
from ploshcha_sim.adapters.tools_fake import FakeToolbox

LATIN_SCHEMA = {
    "type": "object",
    "properties": {"year": {"type": "string"}, "blacksmith": {"type": "string"}},
    "required": ["year", "blacksmith"],
}


def test_our_tool_schemas_are_fully_latin_keyed():
    """Задокументований борг: кожен структурований виклик у системі йде через латинську схему."""
    box = FakeToolbox()
    assert latin_key_share(box.wire_schema()) == 1.0
    assert latin_key_share(box.strict_schema()) == 1.0


def test_schema_keys_reads_both_flat_and_union_shapes():
    box = FakeToolbox()
    assert "tool" in schema_keys(box.wire_schema())
    union_keys = schema_keys(box.strict_schema())
    assert "tool" in union_keys and "entity" in union_keys


def test_latin_key_share_counts_only_pure_latin_keys():
    assert latin_key_share(LATIN_SCHEMA) == 1.0
    assert latin_key_share(ukrainian_schema({"рік": "string", "коваль": "string"})) == 0.0
    mixed = ukrainian_schema({"рік": "string", "year": "string"})
    assert latin_key_share(mixed) == 0.5


def test_ukrainian_schema_shape_matches_what_the_backend_needs():
    schema = ukrainian_schema({"рік": "string", "коваль": "string", "мірошник": "string"})
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["коваль", "мірошник", "рік"], "required обовʼязковий (S1 §14)"
    assert set(schema["properties"]) == {"рік", "коваль", "мірошник"}


def test_ukrainian_schema_respects_explicit_required():
    schema = ukrainian_schema({"рік": "string", "село": "string"}, required=["рік"])
    assert schema["required"] == ["рік"]


def test_value_latin_share_detects_degenerate_output():
    good = '{"рік": "1893", "коваль": "Панас Жмуренко"}'
    bad = '{"year": "location_name_nominative_case_ukrainian_language_version"}'
    assert value_latin_share("Панас Жмуренко 1893") == 0.0
    assert value_latin_share("location_name_nominative_case") == 1.0
    assert value_latin_share(good) < value_latin_share(bad)
