"""Tests for the finalization stage."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai

from ingestion.stages.finalization import (
    _apply_chunk_summary_prefixes,
    _build_chunks,
    _build_degradation_signals,
    _build_document_summary,
    _build_sheet_context_chains,
    _build_sheet_summaries,
    _classify_structure,
    _detect_sections,
    _extract_document_metadata,
    _extract_metrics,
    _finalize_file,
    _generate_document_fields,
    _generate_embeddings,
    _generate_keyword_embeddings,
    _load_enrichment_results,
    _parse_tool_arguments,
    _summarize_sections,
    _summarize_subsections,
    _write_result,
    run_finalization,
)
from ingestion.utils.file_types import (
    DocumentChunk,
    DocumentSection,
    DocumentSubsection,
    FinalizedDocument,
)


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


def _make_section_prompt() -> dict:
    """Build a minimal section-detection prompt for tests."""
    return _make_prompt(
        "File: {filetype}\nPages: {total_pages}\n{page_summaries_json}",
        "detect_sections",
    )


def _make_subsection_prompt() -> dict:
    """Build a minimal subsection-detection prompt for tests."""
    return _make_prompt(
        "File: {filetype}\nSection: {section_title}\n"
        "Pages: {page_start}-{page_end}\n{page_summaries_json}",
        "detect_subsections",
    )


def _make_metadata_prompt() -> dict:
    """Build a minimal metadata-extraction prompt for tests."""
    return _make_prompt(
        "File: {filetype}\nPages: {total_pages}\n{page_summaries_json}",
        "extract_document_metadata",
    )


def _make_structure_prompt() -> dict:
    """Build a minimal structure-classification prompt for tests."""
    return _make_prompt(
        "File: {filetype}\nPages: {total_pages}\n{page_summaries_json}",
        "classify_document_structure",
    )


def _make_summary_prompt() -> dict:
    """Build a minimal structured-summary prompt for tests."""
    return _make_prompt(
        "Title: {section_title}\nPages: {page_start}-{page_end}\n"
        "{page_summaries_json}",
        "generate_section_summary",
    )


def _make_chunk_prompt() -> dict:
    """Build a minimal chunk-summary prompt for tests."""
    return _make_prompt(
        "File: {filetype}\n{chunk_payload_json}",
        "summarize_chunks",
    )


def _make_rollup_prompt() -> dict:
    """Build a minimal document-fields prompt for tests."""
    return _make_prompt(
        "File: {filetype}\nPages: {total_pages}\n{page_summaries_json}",
        "rollup_document",
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


def _make_timeout_error() -> openai.APITimeoutError:
    """Build a retryable OpenAI timeout error for tests."""
    return openai.APITimeoutError(request=httpx.Request("POST", "http://x"))


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


def _make_embedding_response(vectors: list[list[float]]) -> SimpleNamespace:
    """Build an embedding response object."""
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=vector) for vector in vectors]
    )


def test_parse_tool_arguments_valid():
    """Valid tool-call responses decode to a payload dict."""
    payload = {"document_summary": "Summary"}
    parsed = _parse_tool_arguments(_make_tool_response(payload))
    assert parsed == payload


def test_load_enrichment_results(tmp_path, monkeypatch):
    """Loads JSON files from the enrichment directory."""
    enrichment_dir = tmp_path / "processing" / "enrichment"
    enrichment_dir.mkdir(parents=True)
    (enrichment_dir / "report_abc123.json").write_text(
        json.dumps(_make_enrichment()),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "ingestion.stages.finalization.ENRICHMENT_DIR",
        enrichment_dir,
    )

    results = _load_enrichment_results()

    assert len(results) == 1
    assert results[0]["file_path"] == "/data/src/report.pdf"


def test_extract_document_metadata_uses_llm():
    """Metadata extraction normalizes the tool payload."""
    llm = MagicMock()
    llm.call.return_value = _make_tool_response(
        {
            "title": "Q1 Risk Pack",
            "authors": ["Risk Team"],
            "publication_date": "Q1 2026",
            "document_type": "risk report",
            "abstract": "Quarterly risk overview.",
        }
    )

    with patch(
        "ingestion.stages.finalization_support.load_prompt",
        return_value=_make_metadata_prompt(),
    ):
        metadata = _extract_document_metadata(
            pages=[_make_enriched_page()],
            filetype="pdf",
            llm=llm,
            file_label="report.pdf",
        )

    assert metadata["title"] == "Q1 Risk Pack"
    assert metadata["authors"] == ["Risk Team"]
    assert metadata["document_type"] == "risk report"


def test_classify_structure_falls_back_on_error():
    """Structure classification falls back to semantic/low."""
    with (
        patch(
            "ingestion.stages.finalization_support.load_prompt",
            return_value=_make_structure_prompt(),
        ),
        patch(
            "ingestion.stages.finalization_support._call_prompt_with_retry",
            side_effect=RuntimeError("boom"),
        ),
    ):
        structure_type, confidence = _classify_structure(
            pages=[_make_enriched_page()],
            filetype="pdf",
            llm=MagicMock(),
            file_label="report.pdf",
        )

    assert structure_type == "semantic"
    assert confidence == "low"


def test_classify_structure_programmatic_values():
    """Programmatic filetypes skip LLM structure classification."""
    assert _classify_structure([], "pptx", MagicMock(), "deck.pptx") == (
        "slides",
        "high",
    )
    assert _classify_structure([], "xlsx", MagicMock(), "book.xlsx") == (
        "sheets",
        "high",
    )


def test_detect_sections_pdf_with_llm():
    """PDF section detection uses the LLM prompts and range parsing."""
    llm = MagicMock()
    llm.call.side_effect = [
        _make_tool_response(
            {
                "sections": [
                    {
                        "title": "Executive Summary",
                        "page_start": 1,
                        "page_end": 2,
                    }
                ]
            }
        ),
        _make_tool_response(
            {
                "subsections": [
                    {
                        "title": "Market Backdrop",
                        "page_start": 1,
                        "page_end": 1,
                    },
                    {
                        "title": "Key Risks",
                        "page_start": 2,
                        "page_end": 2,
                    },
                ]
            }
        ),
    ]
    pages = [
        _make_enriched_page(
            page_number=1,
            page_title="Page 1",
            summary="Summary 1",
            section_hierarchy=[{"level": 1, "title": "Executive Summary"}],
        ),
        _make_enriched_page(
            page_number=2,
            page_title="Page 2",
            summary="Summary 2",
            section_hierarchy=[{"level": 1, "title": "Executive Summary"}],
        ),
    ]

    with patch(
        "ingestion.stages.finalization_support.load_prompt",
        side_effect=[_make_section_prompt(), _make_subsection_prompt()],
    ):
        sections = _detect_sections(pages, "pdf", llm, "report.pdf")

    assert len(sections) == 1
    assert sections[0].title == "Executive Summary"
    assert sections[0].page_count == 2
    assert [sub.title for sub in sections[0].subsections] == [
        "Market Backdrop",
        "Key Risks",
    ]
    assert llm.call.call_count == 2


def test_detect_sections_pptx_uses_title_slide_boundaries():
    """PPTX sections start on title slides and use slide titles."""
    pages = [
        _make_enriched_page(
            page_number=1,
            page_title="Portfolio Overview",
            metadata={"page_type": "title_slide"},
        ),
        _make_enriched_page(
            page_number=2,
            page_title="Exposure Summary",
            metadata={"page_type": "content_slide"},
        ),
        _make_enriched_page(
            page_number=3,
            page_title="Risk Review",
            metadata={"page_type": "title_slide"},
        ),
        _make_enriched_page(
            page_number=4,
            page_title="Watchlist",
            metadata={"page_type": "content_slide"},
        ),
    ]

    sections = _detect_sections(pages, "pptx", MagicMock(), "deck.pptx")

    assert [section.title for section in sections] == [
        "Portfolio Overview",
        "Risk Review",
    ]
    assert len(sections[0].subsections) == 2
    assert sections[1].subsections[1].title == "Watchlist"


def test_detect_sections_xlsx_uses_sheet_and_dense_table_regions():
    """XLSX sections are per sheet with dense-table subsections."""
    pages = [
        _make_enriched_page(
            page_number=1,
            page_title="Executive Summary",
            metadata={
                "page_type": "page_like_sheet",
                "sheet_name": "Executive Summary",
                "handling_mode": "page_like",
            },
        ),
        _make_enriched_page(
            page_number=2,
            page_title="Loan Tape",
            method="dense_table_replaced",
            metadata={
                "page_type": "dense_table_sheet",
                "sheet_name": "Loan Tape",
                "handling_mode": "dense_table_candidate",
            },
            dense_tables=[
                {
                    "region_id": "region_1",
                    "used_range": "A1:B10",
                    "routing_metadata": {"used_range": "A1:B10"},
                },
                {
                    "region_id": "region_2",
                    "used_range": "C1:D10",
                    "routing_metadata": {"used_range": "C1:D10"},
                },
            ],
        ),
    ]

    sections = _detect_sections(pages, "xlsx", MagicMock(), "book.xlsx")

    assert [section.title for section in sections] == [
        "Executive Summary",
        "Loan Tape",
    ]
    assert sections[0].subsections[0].title == "Executive Summary"
    assert [sub.title for sub in sections[1].subsections] == [
        "Loan Tape [A1:B10]",
        "Loan Tape [C1:D10]",
    ]


def test_summarize_sections_uses_boilerplate_and_llm():
    """Boilerplate sections skip the LLM and use deterministic summaries."""
    llm = MagicMock()
    llm.call.return_value = _make_tool_response(
        {
            "overview": "This section covers the main risk themes.",
            "key_topics": ["credit"],
            "key_metrics": {"CET1": "13.4%"},
            "key_findings": ["Credit quality remained stable."],
            "notable_facts": ["Watchlist inflows improved."],
            "is_fallback": False,
        }
    )
    sections = [
        DocumentSection(
            section_number=1,
            title="References",
            page_start=1,
            page_end=1,
            page_count=1,
        ),
        DocumentSection(
            section_number=2,
            title="Executive Summary",
            page_start=2,
            page_end=2,
            page_count=1,
        ),
    ]
    pages = [
        _make_enriched_page(page_number=1, page_title="References"),
        _make_enriched_page(page_number=2, page_title="Executive Summary"),
    ]

    with patch(
        "ingestion.stages.finalization_support.load_prompt",
        return_value=_make_summary_prompt(),
    ):
        sections = _summarize_sections(
            sections=sections,
            pages=pages,
            filetype="pdf",
            llm=llm,
            file_label="report.pdf",
            document_metadata={"abstract": "Short abstract."},
        )

    assert sections[0].summary["overview"] == "Standard References section"
    assert sections[1].summary["key_metrics"] == {"CET1": "13.4%"}
    assert llm.call.call_count == 1


def test_summarize_subsections_single_page_uses_page_summary():
    """Single-page subsections derive summaries from page enrichment."""
    llm = MagicMock()
    sections = [
        DocumentSection(
            section_number=1,
            title="Executive Summary",
            page_start=1,
            page_end=1,
            page_count=1,
            subsections=[
                DocumentSubsection(
                    subsection_number=1,
                    title="Overview",
                    page_start=1,
                    page_end=1,
                    page_count=1,
                )
            ],
        )
    ]
    pages = [
        _make_enriched_page(
            page_number=1,
            page_title="Overview",
            summary="Quarterly overview page.",
            keywords=["cet1", "credit"],
        )
    ]

    sections = _summarize_subsections(
        sections=sections,
        pages=pages,
        filetype="pdf",
        llm=llm,
        file_label="report.pdf",
        document_metadata={},
    )

    assert sections[0].subsections[0].summary["overview"] == (
        "Quarterly overview page."
    )
    assert sections[0].subsections[0].summary["key_topics"] == [
        "cet1",
        "credit",
    ]
    assert llm.call.call_count == 0


def test_build_document_summary_assembles_metadata_and_sections():
    """Document summary is assembled from metadata and section summaries."""
    summary = _build_document_summary(
        document_metadata={
            "title": "Q1 Risk Pack",
            "authors": ["Risk Team"],
            "publication_date": "Q1 2026",
            "document_type": "risk report",
            "abstract": "Short abstract.",
        },
        sections=[
            DocumentSection(
                section_number=1,
                title="Executive Summary",
                page_start=1,
                page_end=2,
                page_count=2,
                summary={
                    "overview": "Covers the main quarterly trends.",
                    "key_topics": ["credit migration"],
                    "key_metrics": {"CET1": "13.4%"},
                    "key_findings": ["Capital remained strong."],
                    "notable_facts": ["Watchlist inflows improved."],
                    "is_fallback": False,
                },
                subsections=[
                    DocumentSubsection(
                        subsection_number=1,
                        title="Highlights",
                        page_start=1,
                        page_end=1,
                        page_count=1,
                    )
                ],
            )
        ],
        page_count=2,
    )

    assert "# Document Metadata" in summary
    assert "- Title: Q1 Risk Pack" in summary
    assert "**Key Metrics:** CET1 = 13.4%" in summary
    assert "- 1.1 Highlights (page 1)" in summary


def test_generate_document_fields_falls_back_to_title():
    """Description falls back to the title when the rollup output is blank."""
    llm = MagicMock()
    llm.call.return_value = _make_tool_response(
        {
            "document_summary": "Ignored",
            "document_description": "",
            "document_usage": "",
        }
    )

    with patch(
        "ingestion.stages.finalization_support.load_prompt",
        return_value=_make_rollup_prompt(),
    ):
        description, usage, used_fallback = _generate_document_fields(
            document_summary="Assembled summary",
            pages=[_make_enriched_page(page_title="Overview")],
            filetype="pdf",
            llm=llm,
            file_label="report.pdf",
            fallback_title="Q1 Risk Pack",
        )

    assert description == "Q1 Risk Pack"
    assert usage == ""
    assert used_fallback is True


def test_build_chunks_sets_hierarchy_and_dense_table_fields():
    """Chunk building preserves hierarchy tags and dense-table routing."""
    sections = [
        DocumentSection(
            section_number=1,
            title="Overview",
            page_start=1,
            page_end=2,
            page_count=2,
            subsections=[
                DocumentSubsection(
                    subsection_number=1,
                    title="Performance",
                    page_start=1,
                    page_end=1,
                    page_count=1,
                ),
                DocumentSubsection(
                    subsection_number=2,
                    title="Transactions",
                    page_start=2,
                    page_end=2,
                    page_count=1,
                ),
            ],
        )
    ]
    pages = [
        _make_enriched_page(page_number=1, page_title="Performance"),
        _make_enriched_page(
            page_number=2,
            page_title="Transactions",
            method="dense_table_replaced",
            metadata={"sheet_name": "Transactions"},
            dense_tables=[
                {
                    "region_id": "region_2",
                    "routing_metadata": {
                        "sheet_name": "Transactions",
                        "selected_region_id": "region_2",
                    },
                }
            ],
        ),
    ]

    chunks = _build_chunks(pages, sections)

    assert len(chunks) == 2
    assert chunks[0].hierarchy_path == "Overview > Performance"
    assert chunks[0].embedding_prefix == "Overview > Performance: "
    assert chunks[1].is_dense_table_description is True
    assert chunks[1].dense_table_routing["selected_region_id"] == "region_2"


def test_apply_chunk_summary_prefixes_skips_boilerplate():
    """Only non-boilerplate chunks receive LLM summary prefixes."""
    llm = MagicMock()
    llm.call.return_value = _make_tool_response(
        {
            "summaries": [
                {
                    "chunk_number": 0,
                    "summary": "Quarterly risk overview and capital themes.",
                }
            ]
        }
    )
    sections = [
        DocumentSection(
            section_number=1,
            title="Executive Summary",
            page_start=1,
            page_end=1,
            page_count=1,
            summary={"overview": "Quarterly risk overview."},
        ),
        DocumentSection(
            section_number=2,
            title="References",
            page_start=2,
            page_end=2,
            page_count=1,
            summary={"overview": "Standard References section"},
        ),
    ]
    chunks = [
        DocumentChunk(
            chunk_number=0,
            page_number=1,
            content="Overview content",
            primary_section_number=1,
            primary_section_name="Executive Summary",
            subsection_number=0,
            subsection_name="Executive Summary",
            hierarchy_path="Executive Summary",
            primary_section_page_count=1,
            subsection_page_count=1,
            embedding_prefix="Executive Summary: ",
        ),
        DocumentChunk(
            chunk_number=1,
            page_number=2,
            content="References content",
            primary_section_number=2,
            primary_section_name="References",
            subsection_number=0,
            subsection_name="References",
            hierarchy_path="References",
            primary_section_page_count=1,
            subsection_page_count=1,
            embedding_prefix="References: ",
        ),
    ]

    with patch(
        "ingestion.stages.finalization_support.load_prompt",
        return_value=_make_chunk_prompt(),
    ):
        chunks = _apply_chunk_summary_prefixes(
            chunks=chunks,
            sections=sections,
            filetype="pdf",
            llm=llm,
            file_label="report.pdf",
        )

    assert chunks[0].embedding_prefix == (
        "[Quarterly risk overview and capital themes.]\n\n"
    )
    assert chunks[1].embedding_prefix == "References: "


def test_build_sheet_summaries_only_for_xlsx():
    """Sheet summaries are derived from XLSX page summaries."""
    summaries = _build_sheet_summaries(
        pages=[
            _make_enriched_page(
                page_title="Executive Summary",
                metadata={
                    "sheet_name": "Executive Summary",
                    "handling_mode": "page_like",
                },
            )
        ],
        filetype="xlsx",
    )

    assert summaries == [
        {
            "sheet_name": "Executive Summary",
            "summary": "Overview page summary.",
            "usage": "Useful for overview questions.",
            "handling_mode": "page_like",
        }
    ]


def test_build_sheet_context_chains_detects_continuations():
    """Continuation markers create prior-sheet context chains."""
    pages = [
        _make_enriched_page(
            page_number=1,
            page_title="Loan Tape",
            metadata={"sheet_name": "Loan Tape"},
        ),
        _make_enriched_page(
            page_number=2,
            page_title="Loan Tape (continued)",
            summary="Continuation of the prior sheet.",
            metadata={"sheet_name": "Loan Tape (continued)"},
        ),
        _make_enriched_page(
            page_number=3,
            page_title="Loan Tape (continued)",
            summary="Continuation of the prior sheet.",
            metadata={"sheet_name": "Loan Tape (continued)"},
        ),
    ]

    chains = _build_sheet_context_chains(pages, "xlsx")

    assert chains == [
        {
            "sheet_index": 2,
            "sheet_name": "Loan Tape (continued)",
            "context_sheet_indices": [1],
        },
        {
            "sheet_index": 3,
            "sheet_name": "Loan Tape (continued)",
            "context_sheet_indices": [1, 2],
        },
    ]


def test_generate_embeddings_batches_summary_and_chunks(monkeypatch):
    """Embedding generation batches chunks and embeds the summary."""
    monkeypatch.setenv("FINALIZATION_EMBEDDING_BATCH_SIZE", "2")
    llm = MagicMock()
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = [
        _make_embedding_response([[0.1, 0.2]]),
        _make_embedding_response([[1.0], [2.0]]),
        _make_embedding_response([[3.0]]),
    ]
    llm.get_client.return_value = mock_client
    chunks = [
        DocumentChunk(
            chunk_number=0,
            page_number=1,
            content="First page",
            primary_section_number=1,
            primary_section_name="Overview",
            subsection_number=0,
            subsection_name="Overview",
            hierarchy_path="Overview",
            primary_section_page_count=3,
            subsection_page_count=3,
            embedding_prefix="Overview: ",
        ),
        DocumentChunk(
            chunk_number=1,
            page_number=2,
            content="Second page",
            primary_section_number=1,
            primary_section_name="Overview",
            subsection_number=0,
            subsection_name="Overview",
            hierarchy_path="Overview",
            primary_section_page_count=3,
            subsection_page_count=3,
            embedding_prefix="Overview: ",
        ),
        DocumentChunk(
            chunk_number=2,
            page_number=3,
            content="Third page",
            primary_section_number=1,
            primary_section_name="Overview",
            subsection_number=0,
            subsection_name="Overview",
            hierarchy_path="Overview",
            primary_section_page_count=3,
            subsection_page_count=3,
            embedding_prefix="Overview: ",
        ),
    ]

    summary_embedding, embedded_chunks = _generate_embeddings(
        document_summary="Document summary",
        chunks=chunks,
        llm=llm,
        file_label="report.pdf",
    )

    assert summary_embedding == [0.1, 0.2]
    assert [chunk.embedding for chunk in embedded_chunks] == [
        [1.0],
        [2.0],
        [3.0],
    ]
    assert mock_client.embeddings.create.call_count == 3
    assert mock_client.embeddings.create.call_args_list[1].kwargs["input"] == [
        "Overview: First page",
        "Overview: Second page",
    ]


def test_generate_keyword_embeddings_deduplicates_texts():
    """Keyword embeddings reuse one vector for identical contextual text."""
    llm = MagicMock()
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _make_embedding_response(
        [[0.3, 0.4]]
    )
    llm.get_client.return_value = mock_client
    sections = [
        DocumentSection(
            section_number=1,
            title="Overview",
            page_start=1,
            page_end=2,
            page_count=2,
        )
    ]
    pages = [
        _make_enriched_page(
            page_number=1,
            page_title="Overview",
            keywords=["cet1"],
        ),
        _make_enriched_page(
            page_number=2,
            page_title="Overview",
            keywords=["cet1"],
        ),
    ]

    embeddings = _generate_keyword_embeddings(
        pages=pages,
        sections=sections,
        llm=llm,
        file_label="report.pdf",
    )

    assert len(embeddings) == 2
    assert embeddings[0]["embedding"] == [0.3, 0.4]
    assert embeddings[1]["embedding"] == [0.3, 0.4]
    assert mock_client.embeddings.create.call_args.kwargs["input"] == [
        "Overview | keyword: cet1 | section: Overview"
    ]


def test_extract_metrics_embeds_results():
    """Dense-table metric extraction returns embedded metric rows."""
    llm = MagicMock()
    llm.call.return_value = _make_tool_response(
        {
            "metrics": [
                {
                    "metric_name": "CET1 ratio",
                    "platform": "Group",
                    "sub_platform": "Capital",
                    "periods_available": ["Q1 2026"],
                }
            ]
        }
    )
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _make_embedding_response(
        [[0.5, 0.6]]
    )
    llm.get_client.return_value = mock_client
    pages = [
        _make_enriched_page(
            page_number=1,
            page_title="Capital",
            filetype="xlsx",
            metadata={"sheet_name": "Capital"},
            dense_tables=[
                {
                    "used_range": "A1:D25",
                    "region_id": "region_1",
                    "dense_table_eda": {"columns": [{"name": "CET1"}]},
                    "dense_table_description": {
                        "description": "Capital table"
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

    assert metrics == [
        {
            "metric_name": "CET1 ratio",
            "platform": "Group",
            "sub_platform": "Capital",
            "periods_available": ["Q1 2026"],
            "page_number": 1,
            "sheet_name": "Capital",
            "used_range": "A1:D25",
            "region_id": "region_1",
            "embedding": [0.5, 0.6],
        }
    ]


@patch("ingestion.stages.finalization.time.sleep")
def test_generate_embeddings_retries_on_transient_error(mock_sleep):
    """Embedding generation retries retryable failures."""
    llm = MagicMock()
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = [
        _make_timeout_error(),
        _make_embedding_response([[0.1]]),
        _make_embedding_response([[0.2]]),
    ]
    llm.get_client.return_value = mock_client
    chunks = [
        DocumentChunk(
            chunk_number=0,
            page_number=1,
            content="content",
            primary_section_number=1,
            primary_section_name="Overview",
            subsection_number=0,
            subsection_name="Overview",
            hierarchy_path="Overview",
            primary_section_page_count=1,
            subsection_page_count=1,
            embedding_prefix="Overview: ",
        )
    ]

    summary_embedding, embedded_chunks = _generate_embeddings(
        document_summary="Summary",
        chunks=chunks,
        llm=llm,
        file_label="report.pdf",
    )

    assert summary_embedding == [0.1]
    assert embedded_chunks[0].embedding == [0.2]
    mock_sleep.assert_called_once()


def test_build_degradation_signals_collects_expected_conditions():
    """Degradation signals capture the configured quality checks."""
    sections = [
        DocumentSection(
            section_number=1,
            title="Overview",
            page_start=1,
            page_end=1,
            page_count=1,
            summary={"is_fallback": True},
        ),
        DocumentSection(
            section_number=2,
            title="Details",
            page_start=2,
            page_end=2,
            page_count=1,
            summary={"is_fallback": True},
        ),
        DocumentSection(
            section_number=3,
            title="Appendix",
            page_start=3,
            page_end=3,
            page_count=1,
            summary={"is_fallback": False},
        ),
    ]
    pages = [_make_enriched_page(page_number=index) for index in range(1, 7)]

    signals = _build_degradation_signals(
        document_metadata={"title": ""},
        structure_type="semantic",
        structure_confidence="low",
        pages=pages,
        summary_embedding=[],
        sections=sections,
        used_fallback_description=True,
    )

    assert signals == [
        "empty metadata",
        "default structure classification",
        "missing summary embedding",
        "fallback document description",
        "empty section summaries",
    ]


def test_finalize_file_runs_all_steps_in_sequence():
    """Full file finalization wires all processing steps together."""
    sections = [
        DocumentSection(
            section_number=1,
            title="Overview",
            page_start=1,
            page_end=1,
            page_count=1,
        )
    ]
    chunks = [
        DocumentChunk(
            chunk_number=0,
            page_number=1,
            content="content",
            primary_section_number=1,
            primary_section_name="Overview",
            subsection_number=0,
            subsection_name="Overview",
            hierarchy_path="Overview",
            primary_section_page_count=1,
            subsection_page_count=1,
        )
    ]
    enrichment = _make_enrichment(
        file_path="/data/src/workbook.xlsx",
        filetype="xlsx",
        pages=[
            _make_enriched_page(
                page_title="Executive Summary",
                metadata={
                    "sheet_name": "Executive Summary",
                    "handling_mode": "page_like",
                },
            )
        ],
    )

    with (
        patch(
            "ingestion.stages.finalization._extract_document_metadata"
        ) as mock_extract_document_metadata,
        patch(
            "ingestion.stages.finalization._classify_structure"
        ) as mock_classify_structure,
        patch(
            "ingestion.stages.finalization._detect_sections"
        ) as mock_detect_sections,
        patch(
            "ingestion.stages.finalization._summarize_sections"
        ) as mock_summarize_sections,
        patch(
            "ingestion.stages.finalization._summarize_subsections"
        ) as mock_summarize_subsections,
        patch(
            "ingestion.stages.finalization._build_document_summary"
        ) as mock_build_document_summary,
        patch(
            "ingestion.stages.finalization._generate_document_fields"
        ) as mock_generate_document_fields,
        patch(
            "ingestion.stages.finalization._build_sheet_context_chains"
        ) as mock_build_sheet_context_chains,
        patch(
            "ingestion.stages.finalization._extract_metrics"
        ) as mock_extract_metrics,
        patch(
            "ingestion.stages.finalization._build_chunks"
        ) as mock_build_chunks,
        patch(
            "ingestion.stages.finalization._apply_chunk_summary_prefixes"
        ) as mock_apply_chunk_summary_prefixes,
        patch(
            "ingestion.stages.finalization._generate_embeddings"
        ) as mock_generate_embeddings,
        patch(
            "ingestion.stages.finalization._generate_keyword_embeddings"
        ) as mock_generate_keyword_embeddings,
        patch(
            "ingestion.stages.finalization._build_degradation_signals"
        ) as mock_build_degradation_signals,
    ):
        mock_extract_document_metadata.return_value = {"title": "Workbook"}
        mock_classify_structure.return_value = ("sheets", "high")
        mock_detect_sections.return_value = sections
        mock_summarize_sections.return_value = sections
        mock_summarize_subsections.return_value = sections
        mock_build_document_summary.return_value = "Assembled summary"
        mock_generate_document_fields.return_value = (
            "Document description",
            "Document usage",
            False,
        )
        mock_build_sheet_context_chains.return_value = [
            {"sheet_index": 2, "context_sheet_indices": [1]}
        ]
        mock_extract_metrics.return_value = [{"metric_name": "CET1"}]
        mock_build_chunks.return_value = chunks
        mock_apply_chunk_summary_prefixes.return_value = chunks
        mock_generate_embeddings.return_value = ([0.1, 0.2], chunks)
        mock_generate_keyword_embeddings.return_value = [
            {"keyword": "capital"}
        ]
        mock_build_degradation_signals.return_value = []

        result = _finalize_file(enrichment, MagicMock())

    assert result.file_name == "workbook.xlsx"
    assert result.document_metadata == {"title": "Workbook"}
    assert result.structure_type == "sheets"
    assert result.summary_embedding == [0.1, 0.2]
    assert result.keyword_embeddings == [{"keyword": "capital"}]
    assert result.extracted_metrics == [{"metric_name": "CET1"}]
    assert result.sheet_context_chains == [
        {"sheet_index": 2, "context_sheet_indices": [1]}
    ]
    assert result.page_count == 1
    assert result.sheet_summaries[0]["sheet_name"] == "Executive Summary"
    mock_extract_document_metadata.assert_called_once()
    mock_classify_structure.assert_called_once()
    mock_detect_sections.assert_called_once()
    mock_generate_embeddings.assert_called_once()


def test_finalize_file_raises_on_degradation_threshold():
    """Files fail when degradation signals reach the configured threshold."""
    sections = [
        DocumentSection(
            section_number=1,
            title="Overview",
            page_start=1,
            page_end=1,
            page_count=1,
        )
    ]
    chunks = [
        DocumentChunk(
            chunk_number=0,
            page_number=1,
            content="content",
            primary_section_number=1,
            primary_section_name="Overview",
            subsection_number=0,
            subsection_name="Overview",
            hierarchy_path="Overview",
            primary_section_page_count=1,
            subsection_page_count=1,
        )
    ]
    with (
        patch(
            "ingestion.stages.finalization._extract_document_metadata"
        ) as mock_extract_document_metadata,
        patch(
            "ingestion.stages.finalization._classify_structure"
        ) as mock_classify_structure,
        patch(
            "ingestion.stages.finalization._detect_sections"
        ) as mock_detect_sections,
        patch(
            "ingestion.stages.finalization._summarize_sections"
        ) as mock_summarize_sections,
        patch(
            "ingestion.stages.finalization._summarize_subsections"
        ) as mock_summarize_subsections,
        patch(
            "ingestion.stages.finalization._build_document_summary"
        ) as mock_build_document_summary,
        patch(
            "ingestion.stages.finalization._generate_document_fields"
        ) as mock_generate_document_fields,
        patch(
            "ingestion.stages.finalization._build_sheet_context_chains"
        ) as mock_build_sheet_context_chains,
        patch(
            "ingestion.stages.finalization._extract_metrics"
        ) as mock_extract_metrics,
        patch(
            "ingestion.stages.finalization._build_chunks"
        ) as mock_build_chunks,
        patch(
            "ingestion.stages.finalization._apply_chunk_summary_prefixes"
        ) as mock_apply_chunk_summary_prefixes,
        patch(
            "ingestion.stages.finalization._generate_embeddings"
        ) as mock_generate_embeddings,
        patch(
            "ingestion.stages.finalization._generate_keyword_embeddings"
        ) as mock_generate_keyword_embeddings,
        patch(
            "ingestion.stages.finalization._build_degradation_signals"
        ) as mock_build_degradation_signals,
        patch.dict(
            "os.environ",
            {"FINALIZATION_DEGRADATION_SIGNAL_THRESHOLD": "3"},
        ),
    ):
        mock_extract_document_metadata.return_value = {"title": ""}
        mock_classify_structure.return_value = ("semantic", "low")
        mock_detect_sections.return_value = sections
        mock_summarize_sections.return_value = sections
        mock_summarize_subsections.return_value = sections
        mock_build_document_summary.return_value = "Assembled summary"
        mock_generate_document_fields.return_value = ("Overview", "", True)
        mock_build_sheet_context_chains.return_value = []
        mock_extract_metrics.return_value = []
        mock_build_chunks.return_value = chunks
        mock_apply_chunk_summary_prefixes.return_value = chunks
        mock_generate_embeddings.return_value = ([], chunks)
        mock_generate_keyword_embeddings.return_value = []
        mock_build_degradation_signals.return_value = [
            "empty metadata",
            "missing summary embedding",
            "fallback document description",
        ]

        try:
            _finalize_file(_make_enrichment(), MagicMock())
        except RuntimeError as exc:
            assert str(exc) == (
                "Degraded processing: empty metadata, "
                "missing summary embedding, fallback document description"
            )
        else:
            raise AssertionError("Expected RuntimeError")


@patch("ingestion.stages.finalization._write_result")
@patch("ingestion.stages.finalization._finalize_file")
@patch("ingestion.stages.finalization._load_enrichment_results")
def test_run_finalization_parallel_files(
    mock_load,
    mock_finalize_file,
    mock_write,
):
    """Stage orchestration processes all loaded files."""
    mock_load.return_value = [
        _make_enrichment(file_path="/data/a.pdf"),
        _make_enrichment(file_path="/data/b.pdf"),
    ]
    mock_finalize_file.return_value = FinalizedDocument(
        file_path="/data/a.pdf",
        filetype="pdf",
        file_name="a.pdf",
        document_summary="Summary",
        document_description="Description",
        document_usage="Usage",
    )

    run_finalization(MagicMock())

    assert mock_finalize_file.call_count == 2
    assert mock_write.call_count == 2


@patch("ingestion.stages.finalization._write_result")
@patch("ingestion.stages.finalization._finalize_file")
@patch("ingestion.stages.finalization._load_enrichment_results")
def test_run_finalization_skips_failed_files(
    mock_load,
    mock_finalize_file,
    mock_write,
):
    """Stage orchestration logs failures without aborting the stage."""
    mock_load.return_value = [
        _make_enrichment(file_path="/data/a.pdf"),
        _make_enrichment(file_path="/data/b.pdf"),
    ]

    def _finalize(enrichment, _llm):
        if enrichment["file_path"] == "/data/b.pdf":
            raise RuntimeError("boom")
        return FinalizedDocument(
            file_path=enrichment["file_path"],
            filetype="pdf",
            file_name="a.pdf",
            document_summary="Summary",
            document_description="Description",
            document_usage="Usage",
        )

    mock_finalize_file.side_effect = _finalize

    run_finalization(MagicMock())

    assert mock_write.call_count == 1


def test_write_result_includes_derived_counts(tmp_path, monkeypatch):
    """_write_result persists JSON including derived count fields."""
    finalization_dir = tmp_path / "processing" / "finalization"
    monkeypatch.setattr(
        "ingestion.stages.finalization.FINALIZATION_DIR",
        finalization_dir,
    )
    result = FinalizedDocument(
        file_path="/data/src/report.pdf",
        filetype="pdf",
        file_name="report.pdf",
        document_summary="Summary",
        document_description="Description",
        document_usage="Usage",
        sections=[
            DocumentSection(
                section_number=1,
                title="Overview",
                page_start=1,
                page_end=1,
                page_count=1,
            )
        ],
        chunks=[
            DocumentChunk(
                chunk_number=0,
                page_number=1,
                content="content",
                primary_section_number=1,
                primary_section_name="Overview",
                subsection_number=0,
                subsection_name="Overview",
                hierarchy_path="Overview",
                primary_section_page_count=1,
                subsection_page_count=1,
            )
        ],
        dense_tables=[{"region_id": "region_2"}],
    )

    _write_result(result)

    files = list(finalization_dir.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["page_count"] == 1
    assert data["primary_section_count"] == 1
    assert data["chunk_count"] == 1
    assert data["dense_table_count"] == 1
