from typing import Any

from openpyxl.worksheet.worksheet import Worksheet

class Workbook:
    worksheets: list[Worksheet]
    active: Worksheet

    def create_sheet(self, title: str = ...) -> Worksheet: ...
    def save(self, filename: Any) -> None: ...
    def close(self) -> None: ...
