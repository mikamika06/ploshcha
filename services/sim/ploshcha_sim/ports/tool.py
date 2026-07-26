from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

UNKNOWN_TOOL = "unknown_tool"
BAD_ARGS = "bad_args"
NOT_JSON = "not_json"
NO_TOOL_FIELD = "no_tool_field"


class ToolSpec(BaseModel):
    name: str
    description: str
    params: dict


class ToolCall(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool: str
    ok: bool
    value: object | None = None
    error: str | None = None
    latency_ms: int = 0


def _flatten(spec: dict) -> dict:
    if "anyOf" in spec:
        spec = next((s for s in spec["anyOf"] if s.get("type") != "null"), {"type": "string"})
    return {k: v for k, v in spec.items() if k not in ("title", "default")}


def wire_tool_schema(specs: list[ToolSpec]) -> dict:
    props: dict = {"tool": {"type": "string", "enum": [s.name for s in specs]}}
    for s in specs:
        for name, sub in s.params.get("properties", {}).items():
            props.setdefault(name, _flatten(sub))
    return {"type": "object", "properties": props, "required": ["tool"], "additionalProperties": False}


def strict_tool_schema(specs: list[ToolSpec]) -> dict:
    variants = []
    for s in specs:
        props = {"tool": {"const": s.name}}
        for name, sub in s.params.get("properties", {}).items():
            props[name] = _flatten(sub)
        variants.append({
            "type": "object",
            "properties": props,
            "required": sorted(["tool"] + list(s.params.get("required", []))),
            "additionalProperties": False,
        })
    return {"oneOf": variants}


def native_tools(specs: list[ToolSpec]) -> list[dict]:
    return [
        {"type": "function", "function": {"name": s.name, "description": s.description, "parameters": s.params}}
        for s in specs
    ]


def parse_toolcall(data: object, specs: list[ToolSpec]) -> tuple[ToolCall | None, str | None]:
    if not isinstance(data, dict):
        return None, NOT_JSON
    if "tool" not in data:
        return None, NO_TOOL_FIELD
    spec = next((s for s in specs if s.name == data["tool"]), None)
    if spec is None:
        return None, UNKNOWN_TOOL
    allowed = set(spec.params.get("properties", {}))
    args = {k: v for k, v in data.items() if k in allowed}
    for req in spec.params.get("required", []):
        if req not in args:
            return None, BAD_ARGS
    return ToolCall(tool=spec.name, args=args), None


class ToolPort(ABC):
    @abstractmethod
    def specs(self) -> list[ToolSpec]: ...

    @abstractmethod
    def call(self, request: ToolCall) -> ToolResult: ...

    def wire_schema(self) -> dict:
        return wire_tool_schema(self.specs())

    def strict_schema(self) -> dict:
        return strict_tool_schema(self.specs())

    def native(self) -> list[dict]:
        return native_tools(self.specs())

    def parse(self, data: object) -> tuple[ToolCall | None, str | None]:
        return parse_toolcall(data, self.specs())
