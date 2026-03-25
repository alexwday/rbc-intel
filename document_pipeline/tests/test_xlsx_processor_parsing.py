"""Tests for XLSX sheet classification response parsing."""

import json

import pytest

from ingestion.processors.xlsx.processor import _parse_classification_response


@pytest.mark.parametrize(
    ("response", "message"),
    (
        ({}, "choices"),
        ({"choices": [{}]}, "message payload"),
        ({"choices": [{"message": {}}]}, "tool calls"),
        (
            {"choices": [{"message": {"tool_calls": [{}]}}]},
            "function payload",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [{"function": {}}],
                        }
                    }
                ]
            },
            "function arguments",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [{"function": {"arguments": "[]"}}],
                        }
                    }
                ]
            },
            "decode to an object",
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
                                            {
                                                "handling_mode": 3,
                                                "confidence": 0.7,
                                                "rationale": "bad mode",
                                            }
                                        )
                                    }
                                }
                            ],
                        }
                    }
                ]
            },
            "missing handling_mode",
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
                                            {
                                                "handling_mode": "table",
                                                "confidence": 0.7,
                                                "rationale": "bad mode",
                                            }
                                        )
                                    }
                                }
                            ],
                        }
                    }
                ]
            },
            "invalid handling_mode",
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
                                            {
                                                "handling_mode": "page_like",
                                                "confidence": "high",
                                                "rationale": "bad number",
                                            }
                                        )
                                    }
                                }
                            ],
                        }
                    }
                ]
            },
            "numeric confidence",
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
                                            {
                                                "handling_mode": (
                                                    "dense_table_candidate"
                                                ),
                                                "confidence": 0.9,
                                                "rationale": None,
                                            }
                                        )
                                    }
                                }
                            ],
                        }
                    }
                ]
            },
            "non-empty rationale",
        ),
    ),
)
def test_parse_classification_response_rejects_invalid_shapes(
    response, message
):
    """Rejects malformed tool-call responses with clear errors."""
    with pytest.raises(ValueError, match=message):
        _parse_classification_response(response)


def test_parse_classification_response_accepts_legacy_boolean_payload():
    """Legacy boolean payloads still map to the new handling mode."""
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": json.dumps(
                                    {
                                        "contains_dense_table": True,
                                        "confidence": 0.91,
                                        "rationale": "Legacy payload",
                                    }
                                )
                            }
                        }
                    ]
                }
            }
        ]
    }

    parsed = _parse_classification_response(response)

    assert parsed["handling_mode"] == "dense_table_candidate"
    assert parsed["classification"] == "dense_table_candidate"
    assert parsed["contains_dense_table"] is True
