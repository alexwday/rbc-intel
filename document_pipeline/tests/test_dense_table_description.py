"""Tests for dense-table description utility helpers."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ingestion.utils.content_types import (
    ColumnProfile,
    DenseTableDescription,
    TableEDA,
)
from ingestion.utils.dense_table_description import (
    _build_default_column_description,
    _build_deterministic_sample_queries,
    _classify_column_role,
    _describe_dense_table_batched,
    _looks_identifier_name,
    _looks_measure_name,
    _merge_batched_dense_description,
    _normalize_dense_table_description,
    _parse_batch_summary_response,
    batch_columns_for_description,
)


def _make_column(**overrides) -> ColumnProfile:
    """Build a column profile fixture."""
    defaults = {
        "name": "Category",
        "position": "A",
        "dtype": "text",
        "stats": {"value_distribution": {"Open": 2, "Closed": 1}},
        "sample_values": ["Open", "Closed"],
        "non_null_count": 3,
        "null_count": 0,
        "unique_count": 2,
    }
    defaults.update(overrides)
    return ColumnProfile(**defaults)


def _make_eda(columns: list[ColumnProfile]) -> TableEDA:
    """Build a table EDA fixture from columns."""
    return TableEDA(
        row_count=10,
        columns=columns,
        header_row=1,
        framing_context="# Sheet: Test",
        sample_rows=["| 2 | value |"],
        token_count=100,
        used_range="A1:C10",
        header_mode="header_row",
        source_region_id="region_1",
    )


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({}, "choices"),
        ({"choices": [{}]}, "message"),
        ({"choices": [{"message": {}}]}, "tool calls"),
        (
            {"choices": [{"message": {"tool_calls": [{"id": "x"}]}}]},
            "function data",
        ),
        (
            {
                "choices": [
                    {"message": {"tool_calls": [{"function": {"name": "x"}}]}}
                ]
            },
            "arguments",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "arguments": json.dumps(
                                            {"sample_queries": []}
                                        )
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            "description string",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "arguments": json.dumps(
                                            {"description": "ok"}
                                        )
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            "sample_queries",
        ),
    ],
)
def test_parse_batch_summary_response_errors(response, message):
    """Merge response parser raises on malformed payloads."""
    with pytest.raises(ValueError, match=message):
        _parse_batch_summary_response(response)


def test_name_detectors_cover_identifier_and_measure_keywords():
    """Keyword helpers identify common identifier and measure names."""
    assert _looks_identifier_name("Account ID")
    assert not _looks_identifier_name("Category")
    assert _looks_measure_name("Total Amount")
    assert not _looks_measure_name("Narrative")


def test_classify_column_role_covers_remaining_branches():
    """Role classifier handles identifier, mixed measure, and text paths."""
    text_id = _make_column(
        name="Reference Code",
        unique_count=10,
        stats={},
        sample_values=["R1"],
    )
    numeric_id = _make_column(
        name="Account ID",
        position="B",
        dtype="numeric",
        stats={"min": 1.0, "max": 10.0},
        sample_values=["1"],
        unique_count=10,
    )
    mixed_measure = _make_column(
        name="Score Value",
        position="C",
        dtype="mixed",
        stats={},
        sample_values=["1", "High"],
        unique_count=30,
    )
    text_content = _make_column(
        name="Narrative",
        position="D",
        stats={},
        sample_values=["Long note"],
        unique_count=30,
    )

    assert _classify_column_role(text_id, 10) == "identifier"
    assert _classify_column_role(numeric_id, 10) == "identifier"
    assert _classify_column_role(mixed_measure, 40) == "measure"
    assert _classify_column_role(text_content, 40) == "text_content"


def test_build_default_column_description_covers_remaining_branches():
    """Fallback column descriptions cover non-range and text cases."""
    numeric = _make_column(
        name="Amount",
        dtype="numeric",
        stats={},
        sample_values=["10"],
        unique_count=10,
    )
    date = _make_column(
        name="Date",
        position="B",
        dtype="date",
        stats={},
        sample_values=["2024-01-01"],
        unique_count=10,
    )
    boolean = _make_column(
        name="Active",
        position="C",
        dtype="boolean",
        stats={"true_count": 3, "false_count": 7},
        sample_values=["TRUE", "FALSE"],
        unique_count=2,
    )
    sampled_text = _make_column(
        name="Comment",
        position="D",
        stats={},
        sample_values=["Good", "Bad"],
        unique_count=10,
    )
    bare_boolean = _make_column(
        name="Flag",
        position="E",
        dtype="boolean",
        stats={},
        sample_values=["TRUE"],
        unique_count=1,
    )
    plain_text = _make_column(
        name="Blob",
        position="F",
        stats={},
        sample_values=[],
        unique_count=10,
    )

    assert "numeric values for analysis" in _build_default_column_description(
        numeric
    )
    assert "time-based filtering" in _build_default_column_description(date)
    assert "3 true and 7 false" in _build_default_column_description(boolean)
    assert "contains boolean values." in _build_default_column_description(
        bare_boolean
    )
    assert (
        "text values such as Good, Bad"
        in _build_default_column_description(sampled_text)
    )
    assert "contains text values" in _build_default_column_description(
        plain_text
    )


def test_build_deterministic_sample_queries_covers_optional_paths():
    """Fallback sample queries handle missing names, text, and dedupe."""
    eda = _make_eda([_make_column(name="Status")])
    queries = _build_deterministic_sample_queries(
        "Sheet",
        eda,
        {
            "filter": [],
            "identifier": ["Z"],
            "measure": ["Z"],
            "text_content": ["Z"],
        },
    )

    assert any(
        "Which Z values have the highest Z?" in query for query in queries
    )
    assert any("What details are recorded in Z?" in query for query in queries)

    empty_queries = _build_deterministic_sample_queries(
        "Sheet",
        eda,
        {
            "filter": [],
            "identifier": [],
            "measure": [],
            "text_content": [],
        },
    )
    assert empty_queries[0] == "What records are included in Sheet?"
    assert len(empty_queries) >= 3


def test_normalize_dense_table_description_fills_missing_summary_and_queries():
    """Normalization fills missing column descriptions, roles, and summary."""
    eda = _make_eda(
        [
            _make_column(
                name="Category",
                stats={},
                sample_values=["Open"],
                unique_count=2,
            ),
            _make_column(
                name="Date",
                position="B",
                dtype="date",
                stats={
                    "min": "2024-01-01",
                    "max": "2024-12-31",
                },
                sample_values=["2024-01-01"],
                unique_count=10,
            ),
        ]
    )
    description = DenseTableDescription(
        description="",
        column_descriptions=[],
        filter_columns=[],
        identifier_columns=[],
        measure_columns=[],
        text_content_columns=[],
        sample_queries=[],
    )

    normalized = _normalize_dense_table_description("Sheet", eda, description)

    assert "Sheet is a dense table" in normalized.description
    assert normalized.filter_columns == ["A", "B"]
    assert len(normalized.column_descriptions) == 2
    assert normalized.sample_queries


def test_batch_columns_for_description_empty_and_oversized_paths():
    """Batching returns empty for empty tables or unsplittable prompts."""
    empty_eda = _make_eda([])
    assert not batch_columns_for_description(
        "Sheet",
        empty_eda,
        10,
        lambda _title, _eda: "body",
    )

    eda = _make_eda(
        [
            _make_column(name="First"),
            _make_column(name="Second", position="B"),
        ]
    )

    def _estimate(_title, candidate_eda, _builder):
        if (
            len(candidate_eda.columns) == 1
            and candidate_eda.columns[0].name == "First"
        ):
            return 5
        if len(candidate_eda.columns) == 2:
            return 15
        return 20

    with patch(
        "ingestion.utils.dense_table_description."
        "estimate_dense_description_tokens"
    ) as mock_estimate:
        mock_estimate.side_effect = _estimate
        assert not batch_columns_for_description(
            "Sheet", eda, 10, lambda *_: ""
        )

    one_col_eda = _make_eda([_make_column(name="Only")])
    with patch(
        "ingestion.utils.dense_table_description."
        "estimate_dense_description_tokens"
    ) as mock_estimate:
        mock_estimate.return_value = 50
        assert not batch_columns_for_description(
            "Sheet", one_col_eda, 10, lambda *_: ""
        )


def test_merge_batched_dense_description_over_budget_uses_fallback():
    """Merge falls back deterministically when the prompt is oversized."""
    eda = _make_eda([_make_column(name="Amount", dtype="numeric", stats={})])
    batch_description = DenseTableDescription(
        description="Batch summary",
        column_descriptions=[
            {
                "position": "A",
                "name": "Amount",
                "description": "Amount data",
            }
        ],
        filter_columns=[],
        identifier_columns=[],
        measure_columns=["A"],
        text_content_columns=[],
        sample_queries=[],
    )
    llm = MagicMock()

    with (
        patch(
            "ingestion.utils.dense_table_description.load_prompt"
        ) as mock_load_prompt,
        patch(
            "ingestion.utils.dense_table_description._estimate_prompt_tokens"
        ) as mock_tokens,
    ):
        mock_load_prompt.return_value = {
            "stage": "dense_table_description",
            "user_prompt": "Merge",
            "system_prompt": "",
            "tools": [{"type": "function", "function": {}}],
            "tool_choice": "required",
        }
        mock_tokens.return_value = 50000
        merged = _merge_batched_dense_description(
            "Sheet",
            eda,
            batch_description,
            [(eda, batch_description)],
            llm,
        )

    assert "Sheet is a dense table" in merged.description
    llm.call.assert_not_called()


def test_describe_dense_table_batched_single_batch_uses_fallback():
    """Batched description falls back when batching does not split columns."""
    eda = _make_eda([_make_column(name="Amount", dtype="numeric", stats={})])
    llm = MagicMock()

    with patch(
        "ingestion.utils.dense_table_description."
        "batch_columns_for_description"
    ) as mock_batches:
        mock_batches.return_value = [eda.columns]
        description, mode = _describe_dense_table_batched(
            "Sheet",
            eda,
            llm,
            lambda _title, _eda: "body",
            lambda _response: DenseTableDescription(
                description="",
                column_descriptions=[],
                filter_columns=[],
                identifier_columns=[],
                measure_columns=[],
                text_content_columns=[],
                sample_queries=[],
            ),
        )

    assert "Sheet is a dense table" in description.description
    assert mode == "deterministic_fallback"
