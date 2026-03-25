"""
Fiscal Context - RBC fiscal calendar utilities for prompt injection.

Calculates the current fiscal year and quarter based on RBC's November-October
fiscal calendar. Generates XML-formatted context statements injected into agent
prompts via the {{FISCAL_CONTEXT}} placeholder.
"""

from datetime import datetime

QUARTER_RANGES = {
    1: "November 1st to January 31st",
    2: "February 1st to April 30th",
    3: "May 1st to July 31st",
    4: "August 1st to October 31st",
}


def _get_fiscal_period() -> tuple[int, int]:
    """Calculate current fiscal year and quarter from today's date."""
    now = datetime.now()
    fiscal_year = now.year + 1 if now.month >= 11 else now.year
    fiscal_quarter = ((now.month - 11) % 12) // 3 + 1
    return fiscal_year, fiscal_quarter


def generate_fiscal_context_statement() -> str:
    """Build XML block with current date, fiscal year, quarter, and date ranges.

    Returns:
        XML string suitable for injection into system prompts.
    """
    now = datetime.now()
    fy, fq = _get_fiscal_period()
    return (
        "<FISCAL_CONTEXT>\n"
        f"<CURRENT_DATE>{now.strftime('%Y-%m-%d')}</CURRENT_DATE>\n"
        f"<FISCAL_YEAR>{fy} (FY{fy})</FISCAL_YEAR>\n"
        f"<FISCAL_QUARTER>{fq} (Q{fq})</FISCAL_QUARTER>\n"
        f"<QUARTER_RANGE>{QUARTER_RANGES[fq]}</QUARTER_RANGE>\n"
        "<FISCAL_YEAR_DEFINITION>"
        "Our fiscal year runs from November 1st through October 31st."
        "</FISCAL_YEAR_DEFINITION>\n"
        "</FISCAL_CONTEXT>"
    )
