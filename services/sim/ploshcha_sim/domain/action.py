"""Action-space села."""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter


class MoveTo(BaseModel):
    type: Literal["move_to"] = "move_to"
    poi: str = Field(description="id локації-цілі")


class Speak(BaseModel):
    type: Literal["speak"] = "speak"
    to: list[str] = Field(default_factory=list, description="id адресатів; [] = вголос усім поруч")
    text: str = Field(min_length=1)


class UseObject(BaseModel):
    type: Literal["use_object"] = "use_object"
    poi: str


class PostToBoard(BaseModel):
    type: Literal["post_to_board"] = "post_to_board"
    topic: str = Field(min_length=1)


class Reflect(BaseModel):
    type: Literal["reflect"] = "reflect"


class Wait(BaseModel):
    type: Literal["wait"] = "wait"
    reason: str | None = None


Action = Annotated[
    Union[MoveTo, Speak, UseObject, PostToBoard, Reflect, Wait],
    Field(discriminator="type"),
]

ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)

ACTION_TYPES: tuple[str, ...] = (
    "move_to",
    "speak",
    "use_object",
    "post_to_board",
    "reflect",
    "wait",
)


_VARIANTS = (MoveTo, Speak, UseObject, PostToBoard, Reflect, Wait)


def parse_action(data: object) -> Action:
    return ACTION_ADAPTER.validate_python(data)


def action_json_schema() -> dict:
    """Строга схема-union. Для валідації й документації, НЕ для constrained decoding."""
    return ACTION_ADAPTER.json_schema()


def _flatten(spec: dict) -> dict:
    """anyOf[X, null] -> X; прибрати title/default."""
    if "anyOf" in spec:
        spec = next((s for s in spec["anyOf"] if s.get("type") != "null"), {"type": "string"})
    return {k: v for k, v in spec.items() if k not in ("title", "default")}


def wire_action_json_schema() -> dict:
    """Ступінь 1 обмеження: пласка схема — тримає синтаксис і назву дії, поля вільні."""
    props: dict = {"type": {"type": "string", "enum": list(ACTION_TYPES)}}
    for model in _VARIANTS:
        for name, spec in model.model_json_schema().get("properties", {}).items():
            if name != "type":
                props.setdefault(name, _flatten(spec))
    return {
        "type": "object",
        "properties": props,
        "required": ["type"],
        "additionalProperties": False,
    }


def _harden(node: object) -> object:
    """Кожному object дописати required=усі поля + additionalProperties:false.

    Без цього бекенд не компілює граматику: pydantic не кладе поля з `default`
    (а це наш дискримінатор `type`) у `required`.
    """
    if isinstance(node, dict):
        out = {k: _harden(v) for k, v in node.items()}
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = sorted(out["properties"])
        return out
    if isinstance(node, list):
        return [_harden(v) for v in node]
    return node


def strict_action_json_schema() -> dict:
    """Ступінь 2 обмеження: union, посилений під strict — тримає ще й поля під тип дії."""
    return _harden(action_json_schema())  # type: ignore[return-value]
