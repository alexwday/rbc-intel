"""Load prompt definitions from packaged YAML resources."""

from importlib.resources import files
from typing import Any

import yaml

ALLOWED_TOOL_CHOICES = frozenset({"required", "auto", "none"})


def _validate_required_str(
    data: dict[str, Any],
    field_name: str,
    prompt_name: str,
) -> str:
    """Get a required string field or raise.

    Params: data, field_name, prompt_name. Returns: str.
    """
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Prompt '{prompt_name}' requires non-empty string "
            f"field '{field_name}'"
        )
    return value


def _validate_optional_str(
    data: dict[str, Any],
    field_name: str,
    prompt_name: str,
) -> str | None:
    """Get an optional string field or raise.

    Params: data, field_name, prompt_name. Returns: str | None.
    """
    value = data.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"Prompt '{prompt_name}' field '{field_name}' " f"must be a string"
        )
    return value


def _validate_tools(prompt_name: str, tools: Any) -> list | None:
    """Validate tool definitions.

    Params: prompt_name, tools. Returns: list | None.
    """
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise ValueError(
            f"Prompt '{prompt_name}' field 'tools' must be a list"
        )

    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(
                f"Prompt '{prompt_name}' tool #{index} must be a mapping"
            )
        if tool.get("type") != "function":
            raise ValueError(
                f"Prompt '{prompt_name}' tool #{index} must have "
                f"type 'function'"
            )
        function_data = tool.get("function")
        if not isinstance(function_data, dict):
            raise ValueError(
                f"Prompt '{prompt_name}' tool #{index} requires a "
                f"'function' mapping"
            )
        function_name = function_data.get("name")
        if not isinstance(function_name, str) or not function_name.strip():
            raise ValueError(
                f"Prompt '{prompt_name}' tool #{index} requires "
                f"non-empty function.name"
            )
    return tools


def _validate_tool_choice(
    prompt_name: str, tool_choice: Any
) -> str | dict | None:
    """Validate tool_choice.

    Params: prompt_name, tool_choice. Returns: str | dict | None.
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice not in ALLOWED_TOOL_CHOICES:
            raise ValueError(
                f"Prompt '{prompt_name}' field 'tool_choice' must be one "
                f"of {sorted(ALLOWED_TOOL_CHOICES)}"
            )
        return tool_choice
    if not isinstance(tool_choice, dict):
        raise ValueError(
            f"Prompt '{prompt_name}' field 'tool_choice' must be "
            f"a string or mapping"
        )

    if tool_choice.get("type") != "function":
        raise ValueError(
            f"Prompt '{prompt_name}' mapping tool_choice must have "
            f"type 'function'"
        )
    function_data = tool_choice.get("function")
    if not isinstance(function_data, dict):
        raise ValueError(
            f"Prompt '{prompt_name}' mapping tool_choice requires "
            f"a 'function' mapping"
        )
    function_name = function_data.get("name")
    if not isinstance(function_name, str) or not function_name.strip():
        raise ValueError(
            f"Prompt '{prompt_name}' mapping tool_choice requires "
            f"non-empty function.name"
        )
    return tool_choice


def _validate_prompt(name: str, data: Any) -> dict[str, Any]:
    """Validate and normalize prompt data.

    Params: name, data. Returns: dict.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Prompt '{name}' must contain a top-level mapping")

    prompt: dict[str, Any] = {
        "stage": _validate_required_str(data, "stage", name),
        "user_prompt": _validate_required_str(data, "user_prompt", name),
    }

    version = data.get("version")
    if version is not None:
        if not isinstance(version, (str, int, float)):
            raise ValueError(
                f"Prompt '{name}' field 'version' must be a string or number"
            )
        prompt["version"] = str(version)

    description = _validate_optional_str(data, "description", name)
    if description is not None:
        prompt["description"] = description

    system_prompt = _validate_optional_str(data, "system_prompt", name)
    if system_prompt is not None:
        prompt["system_prompt"] = system_prompt

    tools = _validate_tools(name, data.get("tools"))
    tool_choice = _validate_tool_choice(name, data.get("tool_choice"))
    if tools is None and tool_choice not in (None, "none"):
        raise ValueError(
            f"Prompt '{name}' cannot set tool_choice without tools"
        )
    if tools is not None:
        prompt["tools"] = tools
    if tool_choice is not None:
        prompt["tool_choice"] = tool_choice

    return prompt


def load_prompt(name: str) -> dict[str, Any]:
    """Load a prompt definition by filename stem.

    Params:
        name: Prompt filename stem (e.g. "startup_health_check")

    Returns:
        dict with validated keys such as stage, version,
        description, system_prompt, user_prompt, tools,
        and tool_choice

    Example:
        >>> prompt = load_prompt("startup_health_check")
        >>> prompt["stage"]
        "startup"
    """
    resource = files("ingestion").joinpath("prompts", f"{name}.yaml")
    if not resource.is_file():
        raise FileNotFoundError(f"Prompt file not found: {name}.yaml")

    with resource.open("r", encoding="utf-8") as file_handle:
        try:
            data = yaml.safe_load(file_handle)
        except yaml.YAMLError as exc:
            raise ValueError(f"Prompt '{name}' contains invalid YAML") from exc
    return _validate_prompt(name, data)
