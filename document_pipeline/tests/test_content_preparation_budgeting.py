"""Tests for dense-table prompt budgeting and fallback behavior."""

import json
from unittest.mock import MagicMock, patch

from ingestion.stages.content_preparation import (
    _build_deterministic_dense_description,
    _describe_dense_table,
    batch_columns_for_description,
    estimate_dense_description_tokens,
)
from ingestion.utils.content_types import ColumnProfile, TableEDA


def _make_eda(column_count: int = 4) -> TableEDA:
    """Build a dense-table EDA fixture with mixed column types."""
    columns = []
    for index in range(column_count):
        position = chr(ord("A") + index)
        if index == 0:
            columns.append(
                ColumnProfile(
                    name="Date",
                    position=position,
                    dtype="date",
                    stats={
                        "min": "2024-01-01",
                        "max": "2024-12-31",
                    },
                    sample_values=["2024-01-01", "2024-06-30"],
                    non_null_count=10,
                    null_count=0,
                    unique_count=10,
                )
            )
        elif index % 2 == 1:
            columns.append(
                ColumnProfile(
                    name=f"Amount {position}",
                    position=position,
                    dtype="numeric",
                    stats={"min": 10.0, "max": 500.0, "mean": 250.0},
                    sample_values=["10", "500"],
                    non_null_count=10,
                    null_count=0,
                    unique_count=10,
                )
            )
        else:
            columns.append(
                ColumnProfile(
                    name=f"Category {position}",
                    position=position,
                    dtype="text",
                    stats={"value_distribution": {"Open": 6, "Closed": 4}},
                    sample_values=["Open", "Closed"],
                    non_null_count=10,
                    null_count=0,
                    unique_count=2,
                )
            )
    return TableEDA(
        row_count=10,
        columns=columns,
        header_row=1,
        framing_context="# Sheet: Wide",
        sample_rows=["| 2 | 2024-01-01 | 10 | Open |"],
        token_count=500,
        used_range=f"A1:{chr(ord('A') + column_count - 1)}11",
        header_mode="header_row",
        source_region_id="region_1",
    )


def _dense_prompt(name: str) -> dict:
    """Return a mock prompt for dense-table description stages."""
    if name == "dense_table_description":
        return {
            "stage": "dense_table_description",
            "user_prompt": "Describe this table.",
            "system_prompt": "You are an analyst.",
            "tools": [{"type": "function", "function": {}}],
            "tool_choice": "required",
        }
    if name == "dense_table_description_merge":
        return {
            "stage": "dense_table_description",
            "user_prompt": "Summarize these batches.",
            "system_prompt": "You are an analyst.",
            "tools": [{"type": "function", "function": {}}],
            "tool_choice": "required",
        }
    raise AssertionError(f"Unexpected prompt name: {name}")


def _make_description_response(
    columns: list[tuple[str, str]],
    *,
    description: str = "LLM summary.",
    roles: dict[str, list[str]] | None = None,
    sample_queries: list[str] | None = None,
) -> dict:
    """Build a dense-table tool response for tests."""
    role_lists = roles or {}
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": json.dumps(
                                    {
                                        "description": description,
                                        "column_descriptions": [
                                            {
                                                "position": position,
                                                "name": name,
                                                "description": f"{name} data",
                                            }
                                            for position, name in columns
                                        ],
                                        "filter_columns": role_lists.get(
                                            "filter", []
                                        ),
                                        "identifier_columns": role_lists.get(
                                            "identifier", []
                                        ),
                                        "measure_columns": role_lists.get(
                                            "measure", []
                                        ),
                                        "text_content_columns": (
                                            role_lists.get("text_content", [])
                                        ),
                                        "sample_queries": sample_queries
                                        or ["What is the total?"],
                                    }
                                )
                            }
                        }
                    ]
                }
            }
        ]
    }


def _make_merge_response() -> dict:
    """Build a merge-stage summary response."""
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": json.dumps(
                                    {
                                        "description": (
                                            "Merged dataset summary."
                                        ),
                                        "sample_queries": [
                                            "What is the total amount?",
                                            "How does amount vary by date?",
                                        ],
                                    }
                                )
                            }
                        }
                    ]
                }
            }
        ]
    }


def test_estimate_dense_description_tokens():
    """Dense table token estimation returns a positive prompt size."""
    with patch(
        "ingestion.utils.dense_table_description.load_prompt"
    ) as mock_load_prompt:
        mock_load_prompt.side_effect = _dense_prompt

        assert estimate_dense_description_tokens("Wide", _make_eda()) > 0


def test_batch_columns_for_description_preserves_column_order():
    """Column batching keeps original order while enforcing a budget."""

    def _estimate(_page_title, eda, _builder):
        return len(eda.columns) * 100

    with patch(
        "ingestion.utils.dense_table_description."
        "estimate_dense_description_tokens"
    ) as mock_estimate:
        mock_estimate.side_effect = _estimate

        batches = batch_columns_for_description("Wide", _make_eda(5), 250)

        assert [[col.position for col in batch] for batch in batches] == [
            ["A", "B"],
            ["C", "D"],
            ["E"],
        ]


def test_describe_dense_table_uses_one_shot_within_budget():
    """One-shot description is used when the prompt fits."""
    eda = _make_eda(3)
    mock_llm = MagicMock()
    mock_llm.call.return_value = _make_description_response(
        [(col.position, col.name) for col in eda.columns],
        roles={
            "filter": ["A"],
            "measure": ["B"],
            "text_content": ["C"],
        },
    )

    with (
        patch(
            "ingestion.stages.content_preparation._build_dense_table_eda"
        ) as mock_build_eda,
        patch(
            "ingestion.utils.dense_table_description."
            "estimate_dense_description_tokens"
        ) as mock_estimate,
        patch(
            "ingestion.utils.dense_table_description."
            "get_dense_table_description_max_prompt_tokens"
        ) as mock_budget,
        patch(
            "ingestion.utils.dense_table_description.load_prompt"
        ) as mock_load_prompt,
    ):
        mock_build_eda.return_value = eda
        mock_estimate.return_value = 100
        mock_budget.return_value = 400
        mock_load_prompt.side_effect = _dense_prompt

        _, description, mode = _describe_dense_table(
            "Wide", "raw content", mock_llm
        )

    assert description.description == "LLM summary."
    assert mode == "llm_one_shot"
    assert mock_llm.call.call_count == 1


def test_describe_dense_table_batches_when_over_budget():
    """Over-budget dense tables switch to batched description + merge."""
    eda = _make_eda(4)
    mock_llm = MagicMock()
    mock_llm.call.side_effect = [
        _make_description_response(
            [(col.position, col.name) for col in eda.columns[:2]],
            roles={"filter": ["A"], "measure": ["B"]},
        ),
        _make_description_response(
            [(col.position, col.name) for col in eda.columns[2:]],
            roles={"filter": ["C"], "text_content": ["D"]},
        ),
        _make_merge_response(),
    ]

    with (
        patch(
            "ingestion.stages.content_preparation._build_dense_table_eda"
        ) as mock_build_eda,
        patch(
            "ingestion.utils.dense_table_description."
            "estimate_dense_description_tokens"
        ) as mock_estimate,
        patch(
            "ingestion.utils.dense_table_description."
            "batch_columns_for_description"
        ) as mock_batch_columns,
        patch(
            "ingestion.utils.dense_table_description."
            "get_dense_table_description_max_prompt_tokens"
        ) as mock_budget,
        patch(
            "ingestion.utils.dense_table_description._estimate_prompt_tokens"
        ) as mock_merge_tokens,
        patch(
            "ingestion.utils.dense_table_description.load_prompt"
        ) as mock_load_prompt,
    ):
        mock_build_eda.return_value = eda
        mock_estimate.return_value = 500
        mock_batch_columns.return_value = [
            eda.columns[:2],
            eda.columns[2:],
        ]
        mock_budget.return_value = 200
        mock_merge_tokens.return_value = 100
        mock_load_prompt.side_effect = _dense_prompt

        _, description, mode = _describe_dense_table(
            "Wide", "raw content", mock_llm
        )

    assert description.description == "Merged dataset summary."
    assert mode == "llm_batched"
    assert description.filter_columns == ["A", "C"]
    assert description.measure_columns == ["B"]
    assert description.text_content_columns == ["D"]
    assert mock_llm.call.call_count == 3


def test_describe_dense_table_falls_back_to_deterministic_on_batch_failure():
    """Batching failures fall back to deterministic dense-table output."""
    eda = _make_eda(4)
    mock_llm = MagicMock()
    mock_llm.call.side_effect = RuntimeError("prompt too long")

    with (
        patch(
            "ingestion.stages.content_preparation._build_dense_table_eda"
        ) as mock_build_eda,
        patch(
            "ingestion.utils.dense_table_description."
            "estimate_dense_description_tokens"
        ) as mock_estimate,
        patch(
            "ingestion.utils.dense_table_description."
            "batch_columns_for_description"
        ) as mock_batch_columns,
        patch(
            "ingestion.utils.dense_table_description."
            "get_dense_table_description_max_prompt_tokens"
        ) as mock_budget,
        patch(
            "ingestion.utils.dense_table_description.load_prompt"
        ) as mock_load_prompt,
    ):
        mock_build_eda.return_value = eda
        mock_estimate.return_value = 500
        mock_batch_columns.return_value = [
            eda.columns[:2],
            eda.columns[2:],
        ]
        mock_budget.return_value = 200
        mock_load_prompt.side_effect = _dense_prompt

        _, description, mode = _describe_dense_table(
            "Wide", "raw content", mock_llm
        )

    assert "dense table" in description.description
    assert mode == "deterministic_fallback"
    assert len(description.column_descriptions) == 4
    assert description.sample_queries


def test_build_deterministic_dense_description_handles_very_wide_table():
    """Deterministic fallback scales to very wide dense tables."""
    eda = _make_eda(20)

    description = _build_deterministic_dense_description("Wide", eda)

    assert len(description.column_descriptions) == 20
    assert description.measure_columns
    assert len(description.sample_queries) >= 3
