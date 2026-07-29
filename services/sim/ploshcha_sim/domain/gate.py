FINAL_TOOL = "final_answer"


def data_tool_names(specs) -> list[str]:
    return [s.name for s in specs if s.name != FINAL_TOOL]


def needs_loop(specs) -> bool:
    return bool(data_tool_names(specs))


def gate_reason(specs) -> str:
    names = data_tool_names(specs)
    return f"tools:{len(names)}" if names else "no_data_tools"
