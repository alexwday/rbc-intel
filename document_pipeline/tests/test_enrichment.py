"""Tests for the enrichment stage."""

import json
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from ingestion.stages.enrichment import (
    _enrich_file,
    _enrich_page,
    _enrich_page_with_retry,
    _load_content_preparation_results,
    _parse_enrichment_response,
    _write_result,
    run_enrichment,
)
from ingestion.utils.file_types import EnrichedPage, EnrichmentResult


def _make_prompt() -> dict:
    """Build a minimal enrichment prompt for tests."""
    return {
        "stage": "enrichment",
        "system_prompt": "You are an analyst.",
        "user_prompt": (
            "File: {filetype}\n"
            "Page: {page_number}/{total_pages}\n"
            "Title: {page_title}\n"
            "Type: {page_type}\n"
            "{content}"
        ),
        "tool_choice": "required",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "enrich_page",
                    "parameters": {},
                },
            }
        ],
    }


def _make_prepared_page(**overrides) -> dict:
    """Build a prepared page dict for enrichment tests."""
    page = {
        "page_number": 1,
        "page_title": "Overview",
        "content": "# Overview\nRevenue increased year over year.",
        "method": "passthrough",
        "metadata": {"page_type": "narrative"},
        "dense_tables": [],
    }
    page.update(overrides)
    return page


def _make_preparation(**overrides) -> dict:
    """Build a content-preparation result fixture."""
    preparation = {
        "file_path": "/data/src/report.pdf",
        "filetype": "pdf",
        "pages": [_make_prepared_page()],
        "dense_tables_spliced": 0,
    }
    preparation.update(overrides)
    return preparation


def _make_enrichment_response(**overrides) -> dict:
    """Build a tool-calling enrichment response."""
    payload = {
        "summary": "A summary.",
        "usage_description": "Useful for quick review.",
        "keywords": ["revenue", "overview"],
        "classifications": ["narrative"],
        "entities": [{"type": "metric", "value": "Revenue"}],
        "section_hierarchy": [{"level": 1, "title": "Overview"}],
    }
    payload.update(overrides)
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": json.dumps(payload),
                            }
                        }
                    ]
                }
            }
        ]
    }


def _make_timeout_error() -> openai.APITimeoutError:
    """Build a retryable OpenAI timeout error for tests."""
    return openai.APITimeoutError(request=httpx.Request("POST", "http://x"))


def _make_enriched_page(page_number: int, **overrides) -> EnrichedPage:
    """Build an EnrichedPage fixture."""
    page = EnrichedPage(
        page_number=page_number,
        page_title=f"Page {page_number}",
        content=f"content {page_number}",
        method="passthrough",
        summary=f"summary {page_number}",
        usage_description="useful",
        keywords=["keyword"],
        classifications=["narrative"],
        entities=[],
        section_hierarchy=[],
    )
    for key, value in overrides.items():
        setattr(page, key, value)
    return page


def test_parse_enrichment_response_valid():
    """Valid tool-call responses parse into enrichment fields."""
    parsed = _parse_enrichment_response(_make_enrichment_response())
    assert parsed["summary"] == "A summary."
    assert parsed["usage_description"] == "Useful for quick review."
    assert parsed["keywords"] == ["revenue", "overview"]
    assert parsed["classifications"] == ["narrative"]
    assert parsed["entities"][0]["value"] == "Revenue"
    assert parsed["section_hierarchy"][0]["title"] == "Overview"


def test_parse_enrichment_response_missing_tool_calls():
    """Missing tool calls raise ValueError."""
    with pytest.raises(ValueError, match="tool calls"):
        _parse_enrichment_response({"choices": [{"message": {}}]})


def test_load_content_preparation_results(tmp_path, monkeypatch):
    """Loads JSON files from the content-preparation directory."""
    content_prep_dir = tmp_path / "processing" / "content_preparation"
    content_prep_dir.mkdir(parents=True)
    (content_prep_dir / "report_abc123.json").write_text(
        json.dumps(_make_preparation())
    )

    monkeypatch.setattr(
        "ingestion.stages.enrichment.CONTENT_PREP_DIR",
        content_prep_dir,
    )
    results = _load_content_preparation_results()
    assert len(results) == 1
    assert results[0]["file_path"] == "/data/src/report.pdf"


@patch("ingestion.stages.enrichment.load_prompt", return_value=_make_prompt())
def test_enrich_page_valid_response(mock_load_prompt):
    """Enriches a page and preserves dense-table payloads."""
    llm = MagicMock()
    llm.call.return_value = _make_enrichment_response()
    page = _make_prepared_page(
        dense_tables=[
            {
                "region_id": "region_2",
                "used_range": "A3:B5",
                "routing_metadata": {"sheet_name": "Transactions"},
                "raw_content": [
                    {
                        "row_number": 3,
                        "cells": [
                            {"column_number": 1, "value": "2024-01-01"},
                            {"column_number": 2, "value": "5000"},
                        ],
                    }
                ],
                "replacement_content": "# Dense Table: Transactions",
            }
        ]
    )

    result = _enrich_page(page, "xlsx", 3, llm)

    assert result.page_number == 1
    assert result.summary == "A summary."
    assert result.usage_description == "Useful for quick review."
    assert result.keywords == ["revenue", "overview"]
    assert result.classifications == ["narrative"]
    assert result.dense_tables[0].region_id == "region_2"
    user_message = llm.call.call_args.kwargs["messages"][1]["content"]
    assert "File: xlsx" in user_message
    assert "Page: 1/3" in user_message
    assert "Title: Overview" in user_message
    mock_load_prompt.assert_called_once()


@patch("ingestion.stages.enrichment.load_prompt", return_value=_make_prompt())
def test_enrich_page_missing_optional_fields(_mock_load_prompt):
    """Missing list fields default to empty lists."""
    llm = MagicMock()
    llm.call.return_value = _make_enrichment_response(
        keywords=[],
        classifications=[],
        entities=[],
        section_hierarchy=[],
    )

    result = _enrich_page(_make_prepared_page(), "pdf", 1, llm)

    assert result.keywords == []
    assert result.classifications == []
    assert result.entities == []
    assert result.section_hierarchy == []


@patch("ingestion.stages.enrichment.load_prompt", return_value=_make_prompt())
def test_enrich_page_preserves_preparation_fields(_mock_load_prompt):
    """Preparation-only fields are carried into the enriched page."""
    llm = MagicMock()
    llm.call.return_value = _make_enrichment_response()

    result = _enrich_page(
        _make_prepared_page(
            original_content="raw dense table markdown",
            dense_table_eda={"row_count": 3, "used_range": "A3:B5"},
            dense_table_description={"description": "A dense dataset."},
            description_generation_mode="llm_one_shot",
            dense_tables=[
                {
                    "region_id": "region_2",
                    "used_range": "A3:B5",
                    "routing_metadata": {"sheet_name": "Transactions"},
                    "raw_content": [
                        {
                            "row_number": 3,
                            "cells": [
                                {
                                    "column_number": 1,
                                    "value": "2024-01-01",
                                }
                            ],
                        }
                    ],
                    "replacement_content": "# Dense Table: Transactions",
                }
            ],
        ),
        "xlsx",
        1,
        llm,
    )

    assert result.original_content == "raw dense table markdown"
    assert result.dense_table_eda == {"row_count": 3, "used_range": "A3:B5"}
    assert result.dense_table_description == {
        "description": "A dense dataset."
    }
    assert result.description_generation_mode == "llm_one_shot"
    assert result.dense_tables[0].raw_content == [
        {
            "row_number": 3,
            "cells": [{"column_number": 1, "value": "2024-01-01"}],
        }
    ]


@patch("ingestion.stages.enrichment.load_prompt", return_value=_make_prompt())
@patch("ingestion.stages.enrichment.time.sleep")
def test_enrich_page_retry_on_transient_error(mock_sleep, _mock_load_prompt):
    """Retryable enrichment errors are retried before succeeding."""
    llm = MagicMock()
    llm.call.side_effect = [
        _make_timeout_error(),
        _make_enrichment_response(),
    ]

    result = _enrich_page_with_retry(_make_prepared_page(), "pdf", 1, llm)

    assert result.summary == "A summary."
    assert llm.call.call_count == 2
    mock_sleep.assert_called_once()


@patch("ingestion.stages.enrichment.load_prompt", return_value=_make_prompt())
@patch("ingestion.stages.enrichment.time.sleep")
def test_enrich_page_exhausts_retries(
    mock_sleep, _mock_load_prompt, monkeypatch
):
    """Retry exhaustion raises RuntimeError."""
    monkeypatch.setenv("ENRICHMENT_MAX_RETRIES", "2")
    llm = MagicMock()
    llm.call.side_effect = [_make_timeout_error(), _make_timeout_error()]

    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        _enrich_page_with_retry(_make_prepared_page(), "pdf", 1, llm)

    assert llm.call.call_count == 2
    mock_sleep.assert_called_once()


@patch("ingestion.stages.enrichment._enrich_page_with_retry")
def test_enrich_file_sequential_pages(mock_enrich_page):
    """Pages are enriched in page order within a file."""
    seen_pages: list[int] = []

    def _capture(page, _filetype, _total_pages, _llm, file_label=""):
        seen_pages.append(page["page_number"])
        assert file_label == "report.pdf"
        return _make_enriched_page(page["page_number"])

    mock_enrich_page.side_effect = _capture
    preparation = _make_preparation(
        pages=[
            _make_prepared_page(page_number=1),
            _make_prepared_page(page_number=2),
        ]
    )

    result = _enrich_file(preparation, MagicMock())

    assert seen_pages == [1, 2]
    assert result.pages_enriched == 2
    assert result.pages_failed == 0


@patch("ingestion.stages.enrichment._enrich_page_with_retry")
def test_enrich_file_preserves_dense_tables_spliced(mock_enrich_page):
    """File-level dense table splice counts carry into enrichment output."""
    mock_enrich_page.return_value = _make_enriched_page(1)

    result = _enrich_file(
        _make_preparation(dense_tables_spliced=2),
        MagicMock(),
    )

    assert result.dense_tables_spliced == 2


@patch("ingestion.stages.enrichment._enrich_page_with_retry")
def test_enrich_file_page_failure_aborts_file(mock_enrich_page):
    """Any page failure aborts the entire file result."""
    mock_enrich_page.side_effect = [RuntimeError("boom")]

    with pytest.raises(RuntimeError, match="boom"):
        _enrich_file(_make_preparation(), MagicMock())


@patch("ingestion.stages.enrichment._write_result")
@patch("ingestion.stages.enrichment._enrich_file")
@patch("ingestion.stages.enrichment._load_content_preparation_results")
def test_run_enrichment_parallel_files(
    mock_load, mock_enrich_file, mock_write
):
    """Stage orchestration processes all loaded files."""
    mock_load.return_value = [
        _make_preparation(file_path="/data/a.pdf"),
        _make_preparation(file_path="/data/b.pdf"),
    ]
    mock_enrich_file.return_value = EnrichmentResult(
        file_path="/data/a.pdf",
        filetype="pdf",
        pages=[_make_enriched_page(1)],
        pages_enriched=1,
        pages_failed=0,
    )

    run_enrichment(MagicMock())

    assert mock_enrich_file.call_count == 2
    assert mock_write.call_count == 2


@patch("ingestion.stages.enrichment._load_content_preparation_results")
def test_run_enrichment_empty_input(mock_load):
    """Empty input exits the stage early."""
    mock_load.return_value = []
    run_enrichment(MagicMock())


@patch("ingestion.stages.enrichment._enrich_file")
@patch("ingestion.stages.enrichment._load_content_preparation_results")
def test_run_enrichment_writes_json(
    mock_load, mock_enrich_file, tmp_path, monkeypatch
):
    """run_enrichment persists output JSON files."""
    enrichment_dir = tmp_path / "processing" / "enrichment"
    monkeypatch.setattr(
        "ingestion.stages.enrichment.ENRICHMENT_DIR",
        enrichment_dir,
    )
    mock_load.return_value = [_make_preparation()]
    mock_enrich_file.return_value = EnrichmentResult(
        file_path="/data/src/report.pdf",
        filetype="pdf",
        pages=[_make_enriched_page(1)],
        pages_enriched=1,
        pages_failed=0,
    )

    run_enrichment(MagicMock())

    files = list(enrichment_dir.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["file_path"] == "/data/src/report.pdf"


def test_write_result(tmp_path, monkeypatch):
    """_write_result writes a content-addressed JSON file."""
    enrichment_dir = tmp_path / "processing" / "enrichment"
    monkeypatch.setattr(
        "ingestion.stages.enrichment.ENRICHMENT_DIR",
        enrichment_dir,
    )
    result = EnrichmentResult(
        file_path="/data/src/report.pdf",
        filetype="pdf",
        pages=[
            _make_enriched_page(
                1,
                original_content="raw dense table markdown",
            )
        ],
        pages_enriched=1,
        pages_failed=0,
        dense_tables_spliced=2,
    )

    _write_result(result)

    files = list(enrichment_dir.glob("*.json"))
    assert len(files) == 1
    assert files[0].name.startswith("report_")
    data = json.loads(files[0].read_text())
    assert data["dense_tables_spliced"] == 2
    assert data["pages"][0]["original_content"] == "raw dense table markdown"
