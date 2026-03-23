from pathlib import Path

from .workbook.workbook import Workbook

def load_workbook(
    filename: str | Path,
    data_only: bool = ...,
) -> Workbook: ...
