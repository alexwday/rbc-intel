"""XLSX workbook processor."""

from .content_preparation import prepare_xlsx_page
from .processor import process_xlsx

prepare_xlsx_content = prepare_xlsx_page

__all__ = ["process_xlsx", "prepare_xlsx_content", "prepare_xlsx_page"]
