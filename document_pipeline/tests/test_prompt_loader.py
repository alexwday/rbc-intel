"""Tests for ingestion.utils.prompt_loader."""

from pathlib import Path

import pytest

from ingestion.utils import prompt_loader
from ingestion.utils.prompt_loader import load_prompt


def _write_prompt(tmp_path: Path, name: str, contents: str) -> None:
    """Write a prompt YAML file under a temp prompt package."""
    prompts_dir = tmp_path / "ingestion" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / f"{name}.yaml").write_text(contents, encoding="utf-8")


def test_load_prompt_success(monkeypatch, tmp_path):
    """Loads and validates a prompt file."""
    _write_prompt(
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
    monkeypatch.setattr(
        prompt_loader,
        "files",
        lambda _package: tmp_path / "ingestion",
    )

    prompt = load_prompt("sample_prompt")

    assert prompt["stage"] == "startup"
    assert prompt["version"] == "2"
    assert prompt["tool_choice"] == "required"
    assert prompt["tools"][0]["function"]["name"] == "ping"


def test_load_prompt_success_without_tool_choice(monkeypatch, tmp_path):
    """Prompts may omit tool_choice entirely."""
    _write_prompt(
        tmp_path,
        "no_tool_choice",
        """
stage: startup
user_prompt: Respond with status ok
""",
    )
    monkeypatch.setattr(
        prompt_loader,
        "files",
        lambda _package: tmp_path / "ingestion",
    )

    prompt = load_prompt("no_tool_choice")

    assert prompt["stage"] == "startup"
    assert "tool_choice" not in prompt


def test_load_prompt_missing_file(monkeypatch, tmp_path):
    """Missing prompt files raise FileNotFoundError."""
    monkeypatch.setattr(
        prompt_loader,
        "files",
        lambda _package: tmp_path / "ingestion",
    )

    with pytest.raises(FileNotFoundError, match="missing_prompt.yaml"):
        load_prompt("missing_prompt")


def test_load_prompt_invalid_yaml(monkeypatch, tmp_path):
    """Invalid YAML raises a clear ValueError."""
    _write_prompt(tmp_path, "bad_yaml", "stage: [broken")
    monkeypatch.setattr(
        prompt_loader,
        "files",
        lambda _package: tmp_path / "ingestion",
    )

    with pytest.raises(ValueError, match="invalid YAML"):
        load_prompt("bad_yaml")


def test_load_prompt_rejects_non_mapping(monkeypatch, tmp_path):
    """Top-level YAML must be a mapping."""
    _write_prompt(
        tmp_path,
        "bad_shape",
        """
- stage: startup
""",
    )
    monkeypatch.setattr(
        prompt_loader,
        "files",
        lambda _package: tmp_path / "ingestion",
    )

    with pytest.raises(ValueError, match="top-level mapping"):
        load_prompt("bad_shape")


def test_load_prompt_requires_user_prompt(monkeypatch, tmp_path):
    """user_prompt is required."""
    _write_prompt(
        tmp_path,
        "missing_user",
        """
stage: startup
system_prompt: Hello
""",
    )
    monkeypatch.setattr(
        prompt_loader,
        "files",
        lambda _package: tmp_path / "ingestion",
    )

    with pytest.raises(ValueError, match="user_prompt"):
        load_prompt("missing_user")


def test_load_prompt_rejects_non_string_description(monkeypatch, tmp_path):
    """Optional string fields must still be strings."""
    _write_prompt(
        tmp_path,
        "bad_description",
        """
stage: startup
user_prompt: Respond with status ok
description:
  nested: value
""",
    )
    monkeypatch.setattr(
        prompt_loader,
        "files",
        lambda _package: tmp_path / "ingestion",
    )

    with pytest.raises(ValueError, match="description"):
        load_prompt("bad_description")


def test_load_prompt_rejects_tool_choice_without_tools(monkeypatch, tmp_path):
    """tool_choice cannot be set when tools are absent."""
    _write_prompt(
        tmp_path,
        "bad_tool_choice",
        """
stage: startup
user_prompt: Respond with status ok
tool_choice: required
""",
    )
    monkeypatch.setattr(
        prompt_loader,
        "files",
        lambda _package: tmp_path / "ingestion",
    )

    with pytest.raises(ValueError, match="tool_choice without tools"):
        load_prompt("bad_tool_choice")


def test_load_prompt_rejects_invalid_tool_choice(monkeypatch, tmp_path):
    """String tool_choice must be one of the allowed values."""
    _write_prompt(
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
    monkeypatch.setattr(
        prompt_loader,
        "files",
        lambda _package: tmp_path / "ingestion",
    )

    with pytest.raises(ValueError, match="tool_choice"):
        load_prompt("invalid_choice")


def test_load_prompt_rejects_non_list_tools(monkeypatch, tmp_path):
    """tools must be a list."""
    _write_prompt(
        tmp_path,
        "bad_tools",
        """
stage: startup
user_prompt: Respond with status ok
tools: ping
""",
    )
    monkeypatch.setattr(
        prompt_loader,
        "files",
        lambda _package: tmp_path / "ingestion",
    )

    with pytest.raises(ValueError, match="tools"):
        load_prompt("bad_tools")


def test_load_prompt_rejects_non_mapping_tool(monkeypatch, tmp_path):
    """Each tool entry must be a mapping."""
    _write_prompt(
        tmp_path,
        "tool_not_mapping",
        """
stage: startup
user_prompt: Respond with status ok
tools:
  - ping
""",
    )
    monkeypatch.setattr(
        prompt_loader,
        "files",
        lambda _package: tmp_path / "ingestion",
    )

    with pytest.raises(ValueError, match="tool #0"):
        load_prompt("tool_not_mapping")


def test_load_prompt_rejects_non_function_tool(monkeypatch, tmp_path):
    """Tools must declare type=function."""
    _write_prompt(
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
    monkeypatch.setattr(
        prompt_loader,
        "files",
        lambda _package: tmp_path / "ingestion",
    )

    with pytest.raises(ValueError, match="type 'function'"):
        load_prompt("tool_wrong_type")


def test_load_prompt_rejects_missing_function_mapping(monkeypatch, tmp_path):
    """Tools require a function mapping."""
    _write_prompt(
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
    monkeypatch.setattr(
        prompt_loader,
        "files",
        lambda _package: tmp_path / "ingestion",
    )

    with pytest.raises(ValueError, match="'function' mapping"):
        load_prompt("missing_function")


def test_load_prompt_rejects_blank_function_name(monkeypatch, tmp_path):
    """Function tools require a non-empty name."""
    _write_prompt(
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
    monkeypatch.setattr(
        prompt_loader,
        "files",
        lambda _package: tmp_path / "ingestion",
    )

    with pytest.raises(ValueError, match="function.name"):
        load_prompt("blank_function_name")


def test_load_prompt_accepts_mapping_tool_choice(monkeypatch, tmp_path):
    """tool_choice may be a function mapping."""
    _write_prompt(
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
    monkeypatch.setattr(
        prompt_loader,
        "files",
        lambda _package: tmp_path / "ingestion",
    )

    prompt = load_prompt("mapping_tool_choice")

    assert prompt["tool_choice"]["function"]["name"] == "ping"


def test_load_prompt_rejects_non_mapping_tool_choice(monkeypatch, tmp_path):
    """tool_choice must be a string or mapping."""
    _write_prompt(
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
    monkeypatch.setattr(
        prompt_loader,
        "files",
        lambda _package: tmp_path / "ingestion",
    )

    with pytest.raises(ValueError, match="string or mapping"):
        load_prompt("bad_mapping_choice")


def test_load_prompt_rejects_invalid_mapping_tool_choice(
    monkeypatch, tmp_path
):
    """Mapping tool_choice must point at a function name."""
    _write_prompt(
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
    monkeypatch.setattr(
        prompt_loader,
        "files",
        lambda _package: tmp_path / "ingestion",
    )

    with pytest.raises(ValueError, match="function.name"):
        load_prompt("invalid_mapping_choice")


def test_load_prompt_rejects_wrong_mapping_tool_choice_type(
    monkeypatch, tmp_path
):
    """Mapping tool_choice must use type=function."""
    _write_prompt(
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
    monkeypatch.setattr(
        prompt_loader,
        "files",
        lambda _package: tmp_path / "ingestion",
    )

    with pytest.raises(ValueError, match="type 'function'"):
        load_prompt("wrong_mapping_choice_type")


def test_load_prompt_rejects_missing_mapping_tool_choice_function(
    monkeypatch, tmp_path
):
    """Mapping tool_choice requires a function mapping."""
    _write_prompt(
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
    monkeypatch.setattr(
        prompt_loader,
        "files",
        lambda _package: tmp_path / "ingestion",
    )

    with pytest.raises(ValueError, match="'function' mapping"):
        load_prompt("missing_mapping_choice_function")


def test_load_prompt_rejects_invalid_version(monkeypatch, tmp_path):
    """Version must be a string or number if provided."""
    _write_prompt(
        tmp_path,
        "bad_version",
        """
stage: startup
user_prompt: Respond with status ok
version:
  major: 1
""",
    )
    monkeypatch.setattr(
        prompt_loader,
        "files",
        lambda _package: tmp_path / "ingestion",
    )

    with pytest.raises(ValueError, match="version"):
        load_prompt("bad_version")
