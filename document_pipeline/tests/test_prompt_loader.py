"""Tests for ingestion.utils.prompt_loader."""

from pathlib import Path

import pytest

from ingestion.utils.prompt_loader import load_prompt


def _write_prompt(
    tmp_path: Path,
    name: str,
    contents: str,
    prompts_dir: Path | None = None,
) -> Path:
    """Write a prompt YAML file and return the directory."""
    target_dir = prompts_dir or (tmp_path / "prompts")
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / f"{name}.yaml").write_text(contents, encoding="utf-8")
    return target_dir


def test_load_prompt_success(tmp_path):
    """Loads and validates a prompt file."""
    prompts_dir = _write_prompt(
        tmp_path,
        "sample_prompt",
        """
stage: startup
version: 2
description: Sample prompt
system_prompt: Follow the tool contract
user_prompt: Respond with status ok
tool_choice: required
tools:
  - type: function
    function:
      name: ping
      description: Health check
      parameters:
        type: object
        properties:
          status:
            type: string
        required:
          - status
""",
    )

    prompt = load_prompt("sample_prompt", prompts_dir)

    assert prompt["stage"] == "startup"
    assert prompt["version"] == "2"
    assert prompt["tool_choice"] == "required"
    assert prompt["tools"][0]["function"]["name"] == "ping"


def test_load_prompt_success_without_tool_choice(tmp_path):
    """Prompts may omit tool_choice entirely."""
    prompts_dir = _write_prompt(
        tmp_path,
        "no_tool_choice",
        """
stage: startup
user_prompt: Respond with status ok
""",
    )

    prompt = load_prompt("no_tool_choice", prompts_dir)

    assert prompt["stage"] == "startup"
    assert "tool_choice" not in prompt


def test_load_prompt_uses_custom_prompts_dir(tmp_path):
    """A caller may override the prompt directory explicitly."""
    prompts_dir = tmp_path / "processor_prompts"
    _write_prompt(
        tmp_path,
        "page_extraction",
        """
stage: extraction
user_prompt: Extract the page
""",
        prompts_dir=prompts_dir,
    )

    prompt = load_prompt("page_extraction", prompts_dir)

    assert prompt["stage"] == "extraction"


def test_load_prompt_missing_file(tmp_path):
    """Missing prompt files raise FileNotFoundError."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="missing_prompt.yaml"):
        load_prompt("missing_prompt", prompts_dir)


def test_load_prompt_requires_prompts_dir():
    """Omitting prompts_dir raises ValueError."""
    with pytest.raises(ValueError, match="prompts_dir is required"):
        load_prompt("any_prompt")


def test_load_prompt_invalid_yaml(tmp_path):
    """Invalid YAML raises a clear ValueError."""
    prompts_dir = _write_prompt(tmp_path, "bad_yaml", "stage: [broken")

    with pytest.raises(ValueError, match="invalid YAML"):
        load_prompt("bad_yaml", prompts_dir)


def test_load_prompt_rejects_non_mapping(tmp_path):
    """Top-level YAML must be a mapping."""
    prompts_dir = _write_prompt(
        tmp_path,
        "bad_shape",
        """
- stage: startup
""",
    )

    with pytest.raises(ValueError, match="top-level mapping"):
        load_prompt("bad_shape", prompts_dir)


def test_load_prompt_requires_user_prompt(tmp_path):
    """user_prompt is required."""
    prompts_dir = _write_prompt(
        tmp_path,
        "missing_user",
        """
stage: startup
system_prompt: Hello
""",
    )

    with pytest.raises(ValueError, match="user_prompt"):
        load_prompt("missing_user", prompts_dir)


def test_load_prompt_rejects_non_string_description(tmp_path):
    """Optional string fields must still be strings."""
    prompts_dir = _write_prompt(
        tmp_path,
        "bad_description",
        """
stage: startup
user_prompt: Respond with status ok
description:
  nested: value
""",
    )

    with pytest.raises(ValueError, match="description"):
        load_prompt("bad_description", prompts_dir)


def test_load_prompt_rejects_tool_choice_without_tools(tmp_path):
    """tool_choice cannot be set when tools are absent."""
    prompts_dir = _write_prompt(
        tmp_path,
        "bad_tool_choice",
        """
stage: startup
user_prompt: Respond with status ok
tool_choice: required
""",
    )

    with pytest.raises(ValueError, match="tool_choice without tools"):
        load_prompt("bad_tool_choice", prompts_dir)


def test_load_prompt_rejects_invalid_tool_choice(tmp_path):
    """String tool_choice must be one of the allowed values."""
    prompts_dir = _write_prompt(
        tmp_path,
        "invalid_choice",
        """
stage: startup
user_prompt: Respond with status ok
tool_choice: force
tools:
  - type: function
    function:
      name: ping
      parameters:
        type: object
""",
    )

    with pytest.raises(ValueError, match="tool_choice"):
        load_prompt("invalid_choice", prompts_dir)


def test_load_prompt_rejects_non_list_tools(tmp_path):
    """tools must be a list."""
    prompts_dir = _write_prompt(
        tmp_path,
        "bad_tools",
        """
stage: startup
user_prompt: Respond with status ok
tools: ping
""",
    )

    with pytest.raises(ValueError, match="tools"):
        load_prompt("bad_tools", prompts_dir)


def test_load_prompt_rejects_non_mapping_tool(tmp_path):
    """Each tool entry must be a mapping."""
    prompts_dir = _write_prompt(
        tmp_path,
        "tool_not_mapping",
        """
stage: startup
user_prompt: Respond with status ok
tools:
  - ping
""",
    )

    with pytest.raises(ValueError, match="tool #0"):
        load_prompt("tool_not_mapping", prompts_dir)


def test_load_prompt_rejects_non_function_tool(tmp_path):
    """Tools must declare type=function."""
    prompts_dir = _write_prompt(
        tmp_path,
        "tool_wrong_type",
        """
stage: startup
user_prompt: Respond with status ok
tools:
  - type: webhook
    function:
      name: ping
""",
    )

    with pytest.raises(ValueError, match="type 'function'"):
        load_prompt("tool_wrong_type", prompts_dir)


def test_load_prompt_rejects_missing_function_mapping(tmp_path):
    """Tools require a function mapping."""
    prompts_dir = _write_prompt(
        tmp_path,
        "missing_function",
        """
stage: startup
user_prompt: Respond with status ok
tools:
  - type: function
    function: ping
""",
    )

    with pytest.raises(ValueError, match="'function' mapping"):
        load_prompt("missing_function", prompts_dir)


def test_load_prompt_rejects_blank_function_name(tmp_path):
    """Function tools require a non-empty name."""
    prompts_dir = _write_prompt(
        tmp_path,
        "blank_function_name",
        """
stage: startup
user_prompt: Respond with status ok
tools:
  - type: function
    function:
      name: "  "
""",
    )

    with pytest.raises(ValueError, match="function.name"):
        load_prompt("blank_function_name", prompts_dir)


def test_load_prompt_accepts_mapping_tool_choice(tmp_path):
    """tool_choice may be a function mapping."""
    prompts_dir = _write_prompt(
        tmp_path,
        "mapping_tool_choice",
        """
stage: startup
user_prompt: Respond with status ok
tool_choice:
  type: function
  function:
    name: ping
tools:
  - type: function
    function:
      name: ping
      parameters:
        type: object
""",
    )

    prompt = load_prompt("mapping_tool_choice", prompts_dir)

    assert prompt["tool_choice"]["function"]["name"] == "ping"


def test_load_prompt_rejects_non_mapping_tool_choice(tmp_path):
    """tool_choice must be a string or mapping."""
    prompts_dir = _write_prompt(
        tmp_path,
        "bad_mapping_choice",
        """
stage: startup
user_prompt: Respond with status ok
tool_choice:
  - function
tools:
  - type: function
    function:
      name: ping
      parameters:
        type: object
""",
    )

    with pytest.raises(ValueError, match="string or mapping"):
        load_prompt("bad_mapping_choice", prompts_dir)


def test_load_prompt_rejects_invalid_mapping_tool_choice(tmp_path):
    """Mapping tool_choice must point at a function name."""
    prompts_dir = _write_prompt(
        tmp_path,
        "invalid_mapping_choice",
        """
stage: startup
user_prompt: Respond with status ok
tool_choice:
  type: function
  function:
    name: ""
tools:
  - type: function
    function:
      name: ping
      parameters:
        type: object
""",
    )

    with pytest.raises(ValueError, match="function.name"):
        load_prompt("invalid_mapping_choice", prompts_dir)


def test_load_prompt_rejects_wrong_mapping_tool_choice_type(tmp_path):
    """Mapping tool_choice must use type=function."""
    prompts_dir = _write_prompt(
        tmp_path,
        "wrong_mapping_choice_type",
        """
stage: startup
user_prompt: Respond with status ok
tool_choice:
  type: webhook
  function:
    name: ping
tools:
  - type: function
    function:
      name: ping
      parameters:
        type: object
""",
    )

    with pytest.raises(ValueError, match="type 'function'"):
        load_prompt("wrong_mapping_choice_type", prompts_dir)


def test_load_prompt_rejects_missing_mapping_tool_choice_function(tmp_path):
    """Mapping tool_choice requires a function mapping."""
    prompts_dir = _write_prompt(
        tmp_path,
        "missing_mapping_choice_function",
        """
stage: startup
user_prompt: Respond with status ok
tool_choice:
  type: function
  function: ping
tools:
  - type: function
    function:
      name: ping
      parameters:
        type: object
""",
    )

    with pytest.raises(ValueError, match="'function' mapping"):
        load_prompt("missing_mapping_choice_function", prompts_dir)


def test_load_prompt_rejects_invalid_version(tmp_path):
    """Version must be a string or number if provided."""
    prompts_dir = _write_prompt(
        tmp_path,
        "bad_version",
        """
stage: startup
user_prompt: Respond with status ok
version:
  major: 1
""",
    )

    with pytest.raises(ValueError, match="version"):
        load_prompt("bad_version", prompts_dir)
