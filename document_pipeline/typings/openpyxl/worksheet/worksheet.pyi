from typing import Any, Iterator

class Cell:
    row: int
    column: int
    value: Any

class MergedCells:
    ranges: list[Any]

class Worksheet:
    title: str
    merged_cells: MergedCells

    def append(self, iterable: list[Any]) -> None: ...
    def cell(self, row: int, column: int) -> Cell: ...
    def iter_rows(
        self,
        min_row: int = ...,
        max_row: int = ...,
        min_col: int = ...,
        max_col: int = ...,
    ) -> Iterator[tuple[Cell, ...]]: ...
