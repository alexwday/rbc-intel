"""Regression tests for finalization quality fallbacks."""

import json
import zipfile
from unittest.mock import MagicMock, patch

from ingestion.stages.finalization import (
    _extract_document_metadata,
    _extract_metrics,
    _finalize_file,
    _is_formula_metric,
    _summarize_sections,
)
from ingestion.stages.finalization_support import _normalize_summary_dict
from ingestion.utils.file_types import DocumentSection


def _make_prompt(user_prompt: str, function_name: str) -> dict:
    """Build a minimal tool-calling prompt for tests."""
    return {
        "stage": "finalization",
        "system_prompt": "You are a test analyst.",
        "user_prompt": user_prompt,
        "tool_choice": "required",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": function_name,
                    "parameters": {},
                },
            }
        ],
    }


def _make_metadata_prompt() -> dict:
    """Build a minimal metadata-extraction prompt for tests."""
    return _make_prompt(
        "File: {filetype}\nPages: {total_pages}\n{page_summaries_json}",
        "extract_document_metadata",
    )


def _make_summary_prompt() -> dict:
    """Build a minimal structured-summary prompt for tests."""
    return _make_prompt(
        "Title: {section_title}\nPages: {page_start}-{page_end}\n"
        "{page_summaries_json}",
        "generate_section_summary",
    )


def _make_metric_prompt() -> dict:
    """Build a minimal metric-extraction prompt for tests."""
    return _make_prompt(
        "Sheet: {sheet_name}\nRange: {used_range}\n{dense_table_eda_json}",
        "extract_metrics",
    )


def _make_tool_response(payload: dict) -> dict:
    """Build a generic tool-call response."""
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


def _make_enriched_page(**overrides) -> dict:
    """Build an enriched page fixture."""
    page = {
        "page_number": 1,
        "page_title": "Overview",
        "content": "# Overview\nRevenue increased.",
        "method": "passthrough",
        "metadata": {"page_type": "content_page"},
        "summary": "Overview page summary.",
        "usage_description": "Useful for overview questions.",
        "keywords": ["overview"],
        "classifications": ["narrative"],
        "entities": [],
        "section_hierarchy": [{"level": 1, "title": "Overview"}],
        "dense_tables": [],
    }
    page.update(overrides)
    return page


def _make_region_rows(row_values: list[list[tuple[int, str]]]) -> list[dict]:
    """Build region rows from ordered cell tuples."""
    return [
        {
            "row_number": index + 1,
            "cells": [
                {"column_number": column, "value": value}
                for column, value in values
            ],
        }
        for index, values in enumerate(row_values)
    ]


def _make_enrichment(**overrides) -> dict:
    """Build an enrichment result fixture."""
    enrichment = {
        "file_path": "/data/src/report.pdf",
        "filetype": "pdf",
        "pages": [_make_enriched_page()],
        "pages_enriched": 1,
        "pages_failed": 0,
        "dense_tables_spliced": 0,
    }
    enrichment.update(overrides)
    return enrichment


def test_extract_document_metadata_falls_back_to_file_local_dates():
    """Metadata extraction uses page-local date evidence when LLM omits it."""
    llm = MagicMock()
    llm.call.return_value = _make_tool_response(
        {
            "title": "Q1 Risk Pack",
            "authors": ["Risk Team"],
            "publication_date": "",
            "document_type": "risk report",
            "abstract": "Quarterly risk overview.",
        }
    )

    with patch(
        "ingestion.stages.finalization_support.load_prompt",
        return_value=_make_metadata_prompt(),
    ):
        metadata = _extract_document_metadata(
            pages=[
                _make_enriched_page(
                    summary="Quarterly results for Q1 2026.",
                )
            ],
            filetype="docx",
            llm=llm,
            file_label="report.docx",
        )

    assert metadata["publication_date"] == "Q1 2026"


def test_extract_document_metadata_expands_short_docx_title():
    """Short DOCX titles expand with a meaningful secondary heading."""
    llm = MagicMock()
    llm.call.return_value = _make_tool_response(
        {
            "title": "Rapid-risk dashboard",
            "authors": ["Northbridge"],
            "publication_date": "",
            "document_type": "risk dashboard",
            "abstract": "",
        }
    )

    with patch(
        "ingestion.stages.finalization_support.load_prompt",
        return_value=_make_metadata_prompt(),
    ):
        metadata = _extract_document_metadata(
            pages=[
                _make_enriched_page(
                    page_title="Rapid-risk dashboard",
                    summary="A compact KPI dashboard.",
                    section_hierarchy=[
                        {"level": 1, "title": "Rapid-risk dashboard"},
                        {"level": 2, "title": "Table: KPI summary"},
                        {
                            "level": 2,
                            "title": "Northbridge Document Intelligence Flow",
                        },
                    ],
                )
            ],
            filetype="docx",
            llm=llm,
            file_label="dashboard.docx",
        )

    assert metadata["title"] == (
        "Rapid-risk dashboard - Northbridge Document Intelligence Flow"
    )


def test_extract_document_metadata_skips_generic_docx_context_titles():
    """DOCX title expansion ignores generic structural headings."""
    llm = MagicMock()
    llm.call.return_value = _make_tool_response(
        {
            "title": "Rapid-risk dashboard",
            "authors": ["Northbridge"],
            "publication_date": "",
            "document_type": "risk dashboard",
            "abstract": "",
        }
    )

    with patch(
        "ingestion.stages.finalization_support.load_prompt",
        return_value=_make_metadata_prompt(),
    ):
        metadata = _extract_document_metadata(
            pages=[
                _make_enriched_page(
                    page_title="Rapid-risk dashboard",
                    summary="A compact KPI dashboard.",
                    section_hierarchy=[
                        {"level": 1, "title": "Rapid-risk dashboard"},
                        {"level": 2, "title": "KPI cards table"},
                        {"level": 2, "title": "Northbridge Document Flow"},
                    ],
                )
            ],
            filetype="docx",
            llm=llm,
            file_label="dashboard.docx",
        )

    expected_title = "Rapid-risk dashboard - Northbridge Document Flow"
    assert metadata["title"] == expected_title


def test_extract_document_metadata_uses_meaningful_figure_context():
    """DOCX title expansion strips figure prefixes before evaluation."""
    llm = MagicMock()
    llm.call.return_value = _make_tool_response(
        {
            "title": "Rapid-risk dashboard",
            "authors": ["Northbridge"],
            "publication_date": "",
            "document_type": "risk dashboard",
            "abstract": "",
        }
    )

    with patch(
        "ingestion.stages.finalization_support.load_prompt",
        return_value=_make_metadata_prompt(),
    ):
        metadata = _extract_document_metadata(
            pages=[
                _make_enriched_page(
                    page_title="Rapid-risk dashboard",
                    summary="A compact KPI dashboard.",
                    section_hierarchy=[
                        {"level": 1, "title": "Rapid-risk dashboard"},
                        {"level": 2, "title": "Table: Key cards"},
                        {
                            "level": 2,
                            "title": "Figure: Northbridge Document Flow",
                        },
                    ],
                )
            ],
            filetype="docx",
            llm=llm,
            file_label="dashboard.docx",
        )

    expected_title = "Rapid-risk dashboard - Northbridge Document Flow"
    assert metadata["title"] == expected_title


def test_extract_document_metadata_ignores_synthetic_docx_dates(tmp_path):
    """Synthetic python-docx metadata is ignored for publication dates."""
    docx_path = tmp_path / "synthetic.docx"
    with zipfile.ZipFile(docx_path, "w") as archive:
        archive.writestr(
            "docProps/core.xml",
            (
                "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
                "<cp:coreProperties "
                "xmlns:cp='http://schemas.openxmlformats.org/package/2006/"
                "metadata/core-properties' "
                "xmlns:dc='http://purl.org/dc/elements/1.1/' "
                "xmlns:dcterms='http://purl.org/dc/terms/' "
                "xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>"
                "<dc:creator>python-docx</dc:creator>"
                "<dc:description>generated by python-docx</dc:description>"
                "<dcterms:created "
                "xsi:type='dcterms:W3CDTF'>2013-12-23T23:15:00Z"
                "</dcterms:created>"
                "</cp:coreProperties>"
            ),
        )
    llm = MagicMock()
    llm.call.return_value = _make_tool_response(
        {
            "title": "Synthetic doc",
            "authors": ["python-docx"],
            "publication_date": "",
            "document_type": "test doc",
            "abstract": "",
        }
    )

    with patch(
        "ingestion.stages.finalization_support.load_prompt",
        return_value=_make_metadata_prompt(),
    ):
        metadata = _extract_document_metadata(
            pages=[_make_enriched_page(summary="No date here.")],
            filetype="docx",
            llm=llm,
            file_label="synthetic.docx",
            file_path=str(docx_path),
        )

    assert metadata["publication_date"] == ""


def test_extract_document_metadata_keeps_docx_blank_without_evidence():
    """DOCX publication dates remain blank when no local evidence exists."""
    llm = MagicMock()
    llm.call.return_value = _make_tool_response(
        {
            "title": "Risk dashboard",
            "authors": ["Northbridge"],
            "publication_date": "",
            "document_type": "risk dashboard",
            "abstract": "",
        }
    )

    with patch(
        "ingestion.stages.finalization_support.load_prompt",
        return_value=_make_metadata_prompt(),
    ):
        metadata = _extract_document_metadata(
            pages=[_make_enriched_page(summary="No temporal markers.")],
            filetype="docx",
            llm=llm,
            file_label="dashboard.docx",
        )

    assert metadata["publication_date"] == ""


def test_summarize_sections_uses_structured_xlsx_metric_fallback():
    """Page-like XLSX sections derive key metrics from KPI table metadata."""
    llm = MagicMock()
    llm.call.return_value = _make_tool_response(
        {
            "overview": "Executive summary sheet.",
            "key_topics": ["capital"],
            "key_metrics": {},
            "key_findings": [
                "CET1 ratio at 13.4% (up 20 bps QoQ).",
                "Net charge-offs at 0.41% (down 4 bps QoQ).",
            ],
            "notable_facts": [],
            "is_fallback": False,
        }
    )
    pages = [
        _make_enriched_page(
            page_number=1,
            page_title="Executive_Summary",
            metadata={
                "page_type": "page_like_sheet",
                "sheet_name": "Executive_Summary",
                "handling_mode": "page_like",
                "dense_table_regions": [
                    {
                        "rows": _make_region_rows(
                            [
                                [(1, "Northbridge Bank Q1 2026 Risk Pack")],
                                [(1, "Metric"), (2, "Current"), (3, "QoQ")],
                                [
                                    (1, "CET1 ratio"),
                                    (2, "13.4%"),
                                    (3, "+20 bps"),
                                ],
                                [
                                    (1, "Net charge-offs"),
                                    (2, "0.41%"),
                                    (3, "-4 bps"),
                                ],
                                [
                                    (1, "Stage 2 loans"),
                                    (2, "6.8%"),
                                    (3, "+40 bps"),
                                ],
                                [
                                    (1, "Liquidity coverage"),
                                    (2, "139%"),
                                    (3, "+3 pts"),
                                ],
                            ]
                        )
                    }
                ],
            },
        )
    ]
    sections = [
        DocumentSection(
            section_number=1,
            title="Executive_Summary",
            page_start=1,
            page_end=1,
            page_count=1,
        )
    ]

    with patch(
        "ingestion.stages.finalization_support.load_prompt",
        return_value=_make_summary_prompt(),
    ):
        summarized = _summarize_sections(
            sections=sections,
            pages=pages,
            filetype="xlsx",
            llm=llm,
            file_label="book.xlsx",
            document_metadata={},
        )

    assert summarized[0].summary["key_metrics"] == {
        "CET1 ratio": "13.4%",
        "Net charge-offs": "0.41%",
        "Stage 2 loans": "6.8%",
        "Liquidity coverage": "139%",
    }


def test_summarize_sections_uses_structured_dashboard_metric_fallback():
    """Dashboard-style page-like XLSX sheets extract label/value rows."""
    llm = MagicMock()
    llm.call.return_value = _make_tool_response(
        {
            "overview": "Borderline dashboard sheet.",
            "key_topics": ["dashboard"],
            "key_metrics": {},
            "key_findings": [],
            "notable_facts": [],
            "is_fallback": False,
        }
    )
    pages = [
        _make_enriched_page(
            page_number=1,
            page_title="Borderline_Grid",
            metadata={
                "page_type": "page_like_sheet",
                "sheet_name": "Borderline_Grid",
                "handling_mode": "page_like",
                "dense_table_regions": [
                    {
                        "rows": _make_region_rows(
                            [
                                [(1, "Q1 2026 Risk Dashboard")],
                                [
                                    (1, "Metric"),
                                    (2, "Value"),
                                    (3, "Prior"),
                                    (4, "Delta"),
                                ],
                                [
                                    (1, "Credit migration"),
                                    (2, "8.4%"),
                                    (3, "7.8%"),
                                    (4, "+0.6%"),
                                ],
                                [(1, "Narrative note: ignore this row")],
                                [
                                    (1, "Backlog age"),
                                    (2, "24d"),
                                    (3, "18d"),
                                    (4, "+6d"),
                                ],
                                [
                                    (1, "LCR"),
                                    (2, "139%"),
                                    (3, "136%"),
                                    (4, "+3%"),
                                ],
                            ]
                        )
                    }
                ],
            },
        )
    ]
    sections = [
        DocumentSection(
            section_number=1,
            title="Borderline_Grid",
            page_start=1,
            page_end=1,
            page_count=1,
        )
    ]

    with patch(
        "ingestion.stages.finalization_support.load_prompt",
        return_value=_make_summary_prompt(),
    ):
        summarized = _summarize_sections(
            sections=sections,
            pages=pages,
            filetype="xlsx",
            llm=llm,
            file_label="book.xlsx",
            document_metadata={},
        )

    assert summarized[0].summary["key_metrics"] == {
        "Credit migration": "8.4%",
        "Backlog age": "24d",
        "LCR": "139%",
    }


def test_summarize_sections_keeps_metric_fallback_on_summary_failure():
    """Section fallback still extracts structured metrics on LLM failure."""
    llm = MagicMock()
    llm.call.side_effect = RuntimeError("summary failure")
    pages = [
        _make_enriched_page(
            page_number=1,
            page_title="Agenda",
            content=(
                "# Risk dashboard\n"
                "## KPI Cards\n"
                "- Credit migration\n"
                "  - Value: 8.4%\n"
                "  - Note: Watchlist inflows\n"
                "- Fraud losses\n"
                "  - Value: $12M\n"
                "  - Note: Stable quarter\n"
            ),
            metadata={"page_type": "content_slide"},
        )
    ]
    sections = [
        DocumentSection(
            section_number=1,
            title="Agenda",
            page_start=1,
            page_end=1,
            page_count=1,
        )
    ]

    with patch(
        "ingestion.stages.finalization_support.load_prompt",
        return_value=_make_summary_prompt(),
    ):
        summarized = _summarize_sections(
            sections=sections,
            pages=pages,
            filetype="pptx",
            llm=llm,
            file_label="deck.pptx",
            document_metadata={},
        )

    assert summarized[0].summary["is_fallback"] is True
    assert summarized[0].summary["key_metrics"] == {
        "Credit migration": "8.4%",
        "Fraud losses": "$12M",
    }


def test_normalize_summary_dict_rejects_noisy_fallback_metrics():
    """Fallback metric parsing drops chart-series and generic artifacts."""
    summary = _normalize_summary_dict(
        parsed={
            "overview": "Operating view.",
            "key_topics": [],
            "key_metrics": {},
            "key_findings": [
                "Credit migration = 8.4%, Impaired loans = $1.9B.",
                "Response: 19,22,24,28.",
                "Adverse: 11.5%",
                "Severe: 10.7%",
                "Value: $12M",
                "Base scenario ends highest: 13.4%",
                "Retail: 1",
                "Operational incidents = 7.",
                "Formula region: 38 text columns.",
            ],
            "notable_facts": [
                "Visuals: 1",
                "Second dense table: 24",
                "MM range: 1.39",
                "Example rows: 4",
                "Complaints_Log dense table: 60",
                "Populated grid rows: 101",
                "Currency and symbol examples: £14.2m",
                "Total records: 58",
            ],
            "is_fallback": False,
        },
        fallback_overview="Fallback",
        filetype="pptx",
    )

    assert summary["key_metrics"] == {
        "Credit migration": "8.4%",
        "Impaired loans": "$1.9B",
        "Operational incidents": "7",
    }


def test_normalize_summary_dict_uses_pptx_page_content_for_metrics():
    """PPTX slide content is used when summary text omits explicit metrics."""
    summary = _normalize_summary_dict(
        parsed={
            "overview": "Slides cover risk posture and dashboard context.",
            "key_topics": [],
            "key_metrics": {},
            "key_findings": [],
            "notable_facts": [],
            "is_fallback": False,
        },
        fallback_overview="Fallback",
        pages=[
            _make_enriched_page(
                content=(
                    "# Risk dashboard\n"
                    "- Credit migration: 8.4%\n"
                    "- Impaired loans: $1.9B\n"
                    "- Complaints backlog: 118\n"
                )
            )
        ],
        filetype="pptx",
    )

    assert summary["key_metrics"] == {
        "Credit migration": "8.4%",
        "Impaired loans": "$1.9B",
        "Complaints backlog": "118",
    }


def test_normalize_summary_dict_extracts_pptx_kpi_card_values():
    """PPTX KPI-card bullets map labels to the nested `Value:` lines."""
    summary = _normalize_summary_dict(
        parsed={
            "overview": "Slides cover risk posture and dashboard context.",
            "key_topics": [],
            "key_metrics": {},
            "key_findings": [],
            "notable_facts": [],
            "is_fallback": False,
        },
        fallback_overview="Fallback",
        pages=[
            _make_enriched_page(
                content=(
                    "# Risk dashboard\n"
                    "## KPI cards\n"
                    "- Credit migration\n"
                    "  - Value: 8.4%\n"
                    "  - Note: Watchlist inflows\n"
                    "- Fraud losses\n"
                    "  - Value: $12M\n"
                    "  - Note: Stable quarter\n"
                    "- Operational incidents\n"
                    "  - Value: 7\n"
                    "  - Note: Two unresolved\n"
                )
            )
        ],
        filetype="pptx",
    )

    assert summary["key_metrics"] == {
        "Credit migration": "8.4%",
        "Fraud losses": "$12M",
        "Operational incidents": "7",
    }


def test_normalize_summary_dict_filters_generic_llm_key_metrics():
    """Generic LLM key_metrics are dropped before fallback metrics apply."""
    summary = _normalize_summary_dict(
        parsed={
            "overview": "Bridge view.",
            "key_topics": [],
            "key_metrics": {
                "Value": "$12M",
                "Base scenario ends highest": "13.4%",
                "Total records": "58",
            },
            "key_findings": [
                "Credit migration = 8.4%",
                "Impaired loans = $1.9B",
                "Complaints backlog = 118",
            ],
            "notable_facts": [],
            "is_fallback": False,
        },
        fallback_overview="Fallback",
        filetype="pptx",
    )

    assert summary["key_metrics"] == {
        "Credit migration": "8.4%",
        "Impaired loans": "$1.9B",
        "Complaints backlog": "118",
    }


def test_normalize_summary_dict_strips_trailing_metric_qualifiers():
    """Fallback metric labels shed trailing prose qualifiers."""
    summary = _normalize_summary_dict(
        parsed={
            "overview": "Executive summary.",
            "key_topics": [],
            "key_metrics": {},
            "key_findings": [
                "CET1 ratio reported at 13.4%.",
                "Recurring counterparties: 10 main entities.",
            ],
            "notable_facts": [],
            "is_fallback": False,
        },
        fallback_overview="Fallback",
        filetype="pdf",
    )

    assert summary["key_metrics"] == {"CET1 ratio": "13.4%"}


def test_is_formula_metric_matches_renamed_formula_outputs():
    """Formula-scaffolding names are rejected even when renamed."""
    assert _is_formula_metric(
        "Adjusted Ratio (Offset 1) — ROUND((D/E)*100 + 1 + G*10, 2)"
    )
    assert _is_formula_metric("Formula_07")
    assert not _is_formula_metric("Funded_MM")


def test_extract_metrics_skips_formula_only_dense_regions():
    """Formula-only dense tables are skipped before LLM extraction."""
    llm = MagicMock()
    pages = [
        _make_enriched_page(
            page_number=2,
            page_title="Wide_Formulas",
            metadata={"sheet_name": "Wide_Formulas"},
            dense_tables=[
                {
                    "used_range": "K1:AV19",
                    "region_id": "region_2",
                    "dense_table_eda": {
                        "columns": [
                            {"name": "Calc_01"},
                            {"name": "Calc_02"},
                            {"name": "Calc_03"},
                        ]
                    },
                    "dense_table_description": {
                        "measure_columns": [],
                        "column_descriptions": [
                            {
                                "name": "Calc_01",
                                "description": (
                                    "Text of a per-row formula: "
                                    "=ROUND(($Drow/$Erow)*100 + 1, 2)"
                                ),
                            }
                        ],
                    },
                }
            ],
        )
    ]

    with patch(
        "ingestion.stages.finalization_support.load_prompt",
        return_value=_make_metric_prompt(),
    ):
        metrics = _extract_metrics(pages, "xlsx", llm, "book.xlsx")

    assert not metrics
    assert llm.call.call_count == 0


def test_finalize_file_applies_summary_fallbacks_and_formula_region_skip():
    """Integrated finalization keeps useful metrics and skips formulas."""
    enrichment = _make_enrichment(
        file_path="/data/src/refined_core_workbook.xlsx",
        filetype="xlsx",
        pages=[
            _make_enriched_page(
                page_number=1,
                page_title="Executive_Summary",
                summary="Q1 2026 executive summary.",
                metadata={
                    "page_type": "page_like_sheet",
                    "sheet_name": "Executive_Summary",
                    "handling_mode": "page_like",
                    "dense_table_regions": [
                        {
                            "rows": _make_region_rows(
                                [
                                    [
                                        (1, "Metric"),
                                        (2, "Current"),
                                        (3, "QoQ"),
                                    ],
                                    [
                                        (1, "CET1 ratio"),
                                        (2, "13.4%"),
                                        (3, "+20 bps"),
                                    ],
                                    [
                                        (1, "Net charge-offs"),
                                        (2, "0.41%"),
                                        (3, "-4 bps"),
                                    ],
                                    [
                                        (1, "Stage 2 loans"),
                                        (2, "6.8%"),
                                        (3, "+40 bps"),
                                    ],
                                    [
                                        (1, "Liquidity coverage"),
                                        (2, "139%"),
                                        (3, "+3 pts"),
                                    ],
                                ]
                            )
                        }
                    ],
                },
            ),
            _make_enriched_page(
                page_number=2,
                page_title="Wide_Formulas",
                method="dense_table_replaced",
                metadata={
                    "page_type": "dense_table_sheet",
                    "sheet_name": "Wide_Formulas",
                    "handling_mode": "dense_table_candidate",
                },
                dense_tables=[
                    {
                        "used_range": "K1:AV19",
                        "region_id": "region_2",
                        "dense_table_eda": {
                            "columns": [
                                {"name": "Calc_01"},
                                {"name": "Calc_02"},
                                {"name": "Calc_03"},
                            ]
                        },
                        "dense_table_description": {
                            "measure_columns": [],
                            "column_descriptions": [
                                {
                                    "name": "Calc_01",
                                    "description": (
                                        "Formula text "
                                        "=ROUND(($Drow/$Erow)*100 + 1, 2)"
                                    ),
                                }
                            ],
                        },
                    }
                ],
            ),
        ],
    )
    llm = MagicMock()
    llm.call.side_effect = [
        _make_tool_response(
            {
                "title": "Northbridge Bank Q1 2026 Risk & Performance Pack",
                "authors": ["Northbridge Bank"],
                "publication_date": "",
                "document_type": "risk report",
                "abstract": "Quarterly risk pack.",
            }
        ),
        _make_tool_response(
            {
                "overview": "Executive summary sheet.",
                "key_topics": ["capital"],
                "key_metrics": {},
                "key_findings": [],
                "notable_facts": [],
                "is_fallback": False,
            }
        ),
        _make_tool_response(
            {
                "overview": "Formula support sheet.",
                "key_topics": [],
                "key_metrics": {},
                "key_findings": [],
                "notable_facts": [],
                "is_fallback": False,
            }
        ),
    ]

    with (
        patch(
            "ingestion.stages.finalization_support.load_prompt",
            side_effect=[
                _make_metadata_prompt(),
                _make_summary_prompt(),
                _make_summary_prompt(),
                _make_metric_prompt(),
            ],
        ),
        patch(
            "ingestion.stages.finalization._generate_document_fields",
            return_value=("Description", "Usage", False),
        ),
        patch(
            "ingestion.stages.finalization._build_chunks",
            return_value=[],
        ),
        patch(
            "ingestion.stages.finalization._apply_chunk_summary_prefixes",
            return_value=[],
        ),
        patch(
            "ingestion.stages.finalization._generate_embeddings",
            return_value=([], []),
        ),
        patch(
            "ingestion.stages.finalization._generate_keyword_embeddings",
            return_value=[],
        ),
        patch(
            "ingestion.stages.finalization._build_degradation_signals",
            return_value=[],
        ),
    ):
        result = _finalize_file(enrichment, llm)

    assert result.document_metadata["publication_date"] == "Q1 2026"
    assert result.sections[0].summary["key_metrics"] == {
        "CET1 ratio": "13.4%",
        "Net charge-offs": "0.41%",
        "Stage 2 loans": "6.8%",
        "Liquidity coverage": "139%",
    }
    assert not result.extracted_metrics


def test_finalize_file_promotes_docx_structure_to_sections():
    """Detected DOCX section boundaries override a weak topic-based label."""
    enrichment = _make_enrichment(
        file_path="/data/src/refined_adversarial_vision.docx",
        filetype="docx",
        pages=[
            _make_enriched_page(
                page_number=1,
                page_title="Rapid-risk dashboard",
                content=(
                    "# Rapid-risk dashboard\n\n"
                    "### Figure: Northbridge Document Flow\n"
                ),
                section_hierarchy=[
                    {"level": 1, "title": "Rapid-risk dashboard"},
                    {"level": 2, "title": "Figure: Northbridge Document Flow"},
                ],
            ),
            _make_enriched_page(
                page_number=2,
                page_title="Appendix notation and mixed typography",
                content="# Appendix notation and mixed typography",
                section_hierarchy=[
                    {
                        "level": 1,
                        "title": "Appendix notation and mixed typography",
                    }
                ],
            ),
        ],
    )
    llm = MagicMock()
    llm.call.side_effect = [
        _make_tool_response(
            {
                "title": "Rapid-risk dashboard",
                "authors": ["Northbridge"],
                "publication_date": "",
                "document_type": "risk dashboard",
                "abstract": "",
            }
        ),
        _make_tool_response(
            {"structure_type": "topic_based", "confidence": "low"}
        ),
        _make_tool_response(
            {
                "sections": [
                    {"title": "Rapid-risk dashboard", "page_start": 1},
                    {
                        "title": "Appendix notation and mixed typography",
                        "page_start": 2,
                    },
                ]
            }
        ),
        _make_tool_response(
            {
                "overview": "Dashboard page.",
                "key_topics": [],
                "key_metrics": {},
                "key_findings": [],
                "notable_facts": [],
                "is_fallback": False,
            }
        ),
        _make_tool_response(
            {
                "overview": "Appendix page.",
                "key_topics": [],
                "key_metrics": {},
                "key_findings": [],
                "notable_facts": [],
                "is_fallback": False,
            }
        ),
    ]

    with (
        patch(
            "ingestion.stages.finalization_support.load_prompt",
            side_effect=[
                _make_metadata_prompt(),
                _make_prompt(
                    "{page_summaries_json}",
                    "classify_document_structure",
                ),
                _make_prompt(
                    "{page_summaries_json}",
                    "detect_document_sections",
                ),
                _make_summary_prompt(),
                _make_summary_prompt(),
            ],
        ),
        patch(
            "ingestion.stages.finalization._generate_document_fields",
            return_value=("Description", "Usage", False),
        ),
        patch(
            "ingestion.stages.finalization._build_chunks",
            return_value=[],
        ),
        patch(
            "ingestion.stages.finalization._apply_chunk_summary_prefixes",
            return_value=[],
        ),
        patch(
            "ingestion.stages.finalization._generate_embeddings",
            return_value=([], []),
        ),
        patch(
            "ingestion.stages.finalization._generate_keyword_embeddings",
            return_value=[],
        ),
        patch(
            "ingestion.stages.finalization._build_degradation_signals",
            return_value=[],
        ),
    ):
        result = _finalize_file(enrichment, llm)

    assert result.document_metadata["title"] == (
        "Rapid-risk dashboard - Northbridge Document Flow"
    )
    assert result.structure_type == "sections"
    assert result.structure_confidence == "medium"
