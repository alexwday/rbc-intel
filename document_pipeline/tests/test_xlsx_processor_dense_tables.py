"""Dense-table regression tests for the XLSX processor."""

import json
from unittest.mock import MagicMock, patch

from openpyxl import Workbook

from ingestion.processors.xlsx.processor import process_xlsx


def _make_fake_encoding() -> MagicMock:
    """Build a deterministic tokenizer stub for tests."""
    encoding = MagicMock()
    encoding.encode.side_effect = lambda text: [0] * max(1, len(text) // 8)
    return encoding


def _make_prompt() -> dict:
    """Build a minimal XLSX classification prompt for testing."""
    return {
        "stage": "xlsx_classification",
        "system_prompt": "Choose the worksheet handling mode.",
        "user_prompt": "Decide whether the sheet is page_like or dense.",
        "tool_choice": "required",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "classify_xlsx_sheet",
                    "parameters": {},
                },
            }
        ],
    }


def _make_llm_response(
    handling_mode: str,
    confidence: float = 0.9,
    rationale: str = "classification rationale",
) -> dict:
    """Build a minimal tool-calling response payload."""
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": json.dumps(
                                    {
                                        "handling_mode": handling_mode,
                                        "confidence": confidence,
                                        "rationale": rationale,
                                    }
                                )
                            }
                        }
                    ]
                }
            }
        ]
    }


def _save_workbook(path, sheets: list[tuple[str, list[list[object]]]]) -> None:
    """Create a workbook with the provided sheet data."""
    workbook = Workbook()
    first_sheet = workbook.active
    first_sheet.title = sheets[0][0]
    for row in sheets[0][1]:
        first_sheet.append(row)

    for title, rows in sheets[1:]:
        sheet = workbook.create_sheet(title=title)
        for row in rows:
            sheet.append(row)

    workbook.save(path)
    workbook.close()


@patch(
    "ingestion.processors.xlsx.processor.get_stage_model_config",
    return_value={
        "model": "gpt-5-mini",
        "max_tokens": 800,
        "temperature": None,
    },
)
@patch(
    "ingestion.processors.xlsx.processor.tiktoken.encoding_for_model",
    return_value=_make_fake_encoding(),
)
@patch(
    "ingestion.processors.xlsx.processor.get_xlsx_sheet_token_limit",
    return_value=12000,
)
@patch(
    "ingestion.processors.xlsx.processor.load_prompt",
    return_value=_make_prompt(),
)
def test_process_xlsx_dense_numeric_sheet_fits_inline_but_stays_dense(
    _prompt, _token_limit, _encoding, _model_config, tmp_path
):
    """Inline numeric ledgers still go through dense-table classification."""
    file_path = tmp_path / "dense_numeric.xlsx"
    rows = [
        [
            "Account",
            "Region",
            "Quarter",
            "Product",
            "Amount",
            "Balance",
            "Variance",
            "Status",
            "Owner",
            "RiskBand",
        ]
    ]
    for index in range(1, 61):
        rows.append(
            [
                f"Account {index}",
                f"Region {index % 6}",
                f"Q{(index % 4) + 1}",
                f"Product {index % 8}",
                index * 1200,
                index * 875,
                index - 12,
                "Open" if index % 2 else "Closed",
                f"Analyst {index % 7}",
                f"Band {(index % 5) + 1}",
            ]
        )
    _save_workbook(file_path, [("Transactions", rows)])
    llm = MagicMock()
    llm.call.return_value = _make_llm_response(
        "dense_table_candidate",
        0.96,
        "This is a row-level transaction ledger.",
    )

    result = process_xlsx(str(file_path), llm)

    page = result.pages[0]
    assert page.metadata["threshold_exceeded"] is False
    assert page.metadata["classification"] == "dense_table_candidate"
    assert page.metadata["contains_dense_table"] is True
    llm.call.assert_called_once()


@patch(
    "ingestion.processors.xlsx.processor.get_stage_model_config",
    return_value={
        "model": "gpt-5-mini",
        "max_tokens": 800,
        "temperature": None,
    },
)
@patch(
    "ingestion.processors.xlsx.processor.tiktoken.encoding_for_model",
    return_value=_make_fake_encoding(),
)
@patch(
    "ingestion.processors.xlsx.processor.get_xlsx_sheet_token_limit",
    return_value=12000,
)
@patch(
    "ingestion.processors.xlsx.processor.load_prompt",
    return_value=_make_prompt(),
)
def test_process_xlsx_text_heavy_record_sheet_stays_dense_candidate(
    _prompt, _token_limit, _encoding, _model_config, tmp_path
):
    """Text-heavy row records do not trip the page-like shortcut."""
    file_path = tmp_path / "customer_complaints.xlsx"
    rows = [
        [
            "CaseID",
            "DateOpened",
            "CustomerName",
            "Product",
            "Channel",
            "ComplaintDescription",
            "Resolution",
            "Status",
            "AssignedTo",
        ]
    ]
    for index in range(1, 41):
        rows.append(
            [
                f"C-{index:03d}",
                f"2026-03-{(index % 28) + 1:02d}",
                f"Customer {index}",
                f"Product {index % 9}",
                "Email" if index % 2 else "Phone",
                (
                    "Customer described repeated defects, inconsistent "
                    "packaging quality, delayed follow-up, and missing "
                    "replacement components across multiple orders."
                ),
                (
                    "Support reviewed the shipment history, coordinated with "
                    "the store team, arranged a replacement order, and "
                    "issued a courtesy credit to resolve the complaint."
                ),
                "Resolved" if index % 3 else "Escalated",
                f"Agent {index % 6}",
            ]
        )
    _save_workbook(file_path, [("CustomerComplaints", rows)])
    llm = MagicMock()
    llm.call.return_value = _make_llm_response(
        "dense_table_candidate",
        0.95,
        "This is a row-level case management dataset.",
    )

    result = process_xlsx(str(file_path), llm)

    page = result.pages[0]
    assert page.metadata["classification"] == "dense_table_candidate"
    assert page.metadata["contains_dense_table"] is True
    llm.call.assert_called_once()
