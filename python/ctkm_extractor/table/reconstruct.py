"""Mô hình dữ liệu bảng + fallback dựng bảng bằng cách cluster bounding box.

Đây là tầng fallback thứ 2 (sau PP-Structure): gom token thành hàng theo trục y,
rồi gom thành cột theo trục x. Cách này đơn giản, không cần model, nhưng dễ sai
với ô merge dọc - vì vậy PP-Structure vẫn được ưu tiên.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

from ..ocr.base import BoundingBox, OCRToken, group_tokens_into_lines

logger = logging.getLogger(__name__)

#: Token có bề rộng lớn hơn tỉ lệ này so với bảng thì không dùng để dựng biên cột
#: (thường là tiêu đề bảng hoặc ô merge ngang).
WIDE_TOKEN_RATIO = 0.45

#: Ngưỡng tối thiểu phần diện tích token nằm trong ô để coi là thuộc ô đó.
MIN_CELL_OVERLAP = 0.30


@dataclass
class TableCell:
    """Một ô của bảng, có hỗ trợ ô gộp (``row_span``/``col_span``)."""

    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    box: BoundingBox | None = None
    text: str = ""
    tokens: list[OCRToken] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.row_span = max(1, int(self.row_span))
        self.col_span = max(1, int(self.col_span))
        self.text = (self.text or "").strip()

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def occupies(self) -> Iterator[tuple[int, int]]:
        """Sinh toàn bộ toạ độ ``(row, col)`` mà ô này chiếm."""
        for r in range(self.row, self.row + self.row_span):
            for c in range(self.col, self.col + self.col_span):
                yield (r, c)

    def covers(self, row: int, col: int) -> bool:
        return (
            self.row <= row < self.row + self.row_span
            and self.col <= col < self.col + self.col_span
        )


@dataclass
class Table:
    """Bảng đã dựng: danh sách ô + kích thước lưới."""

    cells: list[TableCell] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    #: Nguồn dựng bảng: ``pp_structure`` | ``cluster`` | ``manual``.
    source: str = "unknown"
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.cells:
            self.n_rows = max(self.n_rows, max(c.row + c.row_span for c in self.cells))
            self.n_cols = max(self.n_cols, max(c.col + c.col_span for c in self.cells))

    @property
    def is_empty(self) -> bool:
        return not any(not cell.is_empty for cell in self.cells)

    def cell_at(self, row: int, col: int) -> TableCell | None:
        """Ô phủ toạ độ ``(row, col)``; ô gộp trả về chính nó ở mọi toạ độ nó phủ."""
        for cell in self.cells:
            if cell.covers(row, col):
                return cell
        return None

    def row_cells(self, row: int) -> list[TableCell]:
        """Các ô thuộc một hàng, sắp theo cột (ô gộp dọc xuất hiện ở mọi hàng nó phủ)."""
        found: list[TableCell] = [c for c in self.cells if c.row <= row < c.row + c.row_span]
        return sorted(found, key=lambda c: c.col)

    def column_cells(self, col: int) -> list[TableCell]:
        """Các ô thuộc một cột, sắp theo hàng."""
        found: list[TableCell] = [c for c in self.cells if c.col <= col < c.col + c.col_span]
        return sorted(found, key=lambda c: c.row)

    def iter_cells(self) -> Iterator[TableCell]:
        yield from sorted(self.cells, key=lambda c: (c.row, c.col))

    def to_grid(self) -> list[list[str]]:
        """Lưới text đầy đủ; ô gộp lặp lại text ở mọi vị trí nó phủ."""
        grid = [["" for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        for cell in self.cells:
            for row, col in cell.occupies():
                if 0 <= row < self.n_rows and 0 <= col < self.n_cols:
                    grid[row][col] = cell.text
        return grid

    def render(self, max_width: int = 40) -> str:
        """Chuỗi dạng bảng ASCII để in khi chạy ``--debug``."""
        grid = self.to_grid()
        if not grid:
            return "(bảng rỗng)"
        widths = [0] * self.n_cols
        for row in grid:
            for index, value in enumerate(row):
                widths[index] = min(max_width, max(widths[index], len(value)))
        lines: list[str] = []
        separator = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        lines.append(separator)
        for row in grid:
            cells = []
            for index, value in enumerate(row):
                shown = value if len(value) <= widths[index] else value[: widths[index] - 1] + "…"
                cells.append(" " + shown.ljust(widths[index]) + " ")
            lines.append("|" + "|".join(cells) + "|")
            lines.append(separator)
        return "\n".join(lines)


def assign_tokens_to_cells(
    cells: Sequence[TableCell],
    tokens: Iterable[OCRToken],
    min_overlap: float = MIN_CELL_OVERLAP,
) -> int:
    """Gán token OCR vào ô theo diện tích chồng lấn; trả số token gán được.

    Dùng khi cấu trúc ô đến từ PP-Structure (chỉ có toạ độ) còn text đến từ
    VietOCR - hai nguồn được ghép lại bằng toạ độ.
    """
    boxed = [cell for cell in cells if cell.box is not None]
    assigned = 0
    for token in tokens:
        if token.is_empty:
            continue
        best: TableCell | None = None
        best_score = 0.0
        for cell in boxed:
            assert cell.box is not None
            score = token.box.overlap_ratio(cell.box)
            if score > best_score:
                best_score = score
                best = cell
        if best is None or best_score < min_overlap:
            logger.debug("Token %r không khớp ô nào (score=%.2f)", token.text, best_score)
            continue
        best.tokens.append(token)
        assigned += 1

    for cell in boxed:
        if cell.tokens:
            cell.text = join_tokens(cell.tokens)
    return assigned


def join_tokens(tokens: Sequence[OCRToken]) -> str:
    """Ghép text các token trong một ô theo thứ tự đọc, mỗi dòng cách nhau 1 space."""
    lines = group_tokens_into_lines(list(tokens))
    parts = [" ".join(tok.text for tok in line if tok.text) for line in lines]
    return " ".join(part for part in parts if part).strip()


def _merge_intervals(
    intervals: Sequence[tuple[float, float]], min_gap: float
) -> list[tuple[float, float]]:
    """Gộp các khoảng [x1, x2] chồng lấn hoặc cách nhau dưới ``min_gap``."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda i: i[0])
    merged: list[list[float]] = [list(ordered[0])]
    for start, end in ordered[1:]:
        current = merged[-1]
        if start <= current[1] + min_gap:
            current[1] = max(current[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def detect_column_bands(
    tokens: Sequence[OCRToken], min_gap: float | None = None
) -> list[tuple[float, float]]:
    """Suy ra biên các cột từ hình chiếu bounding box lên trục x."""
    if not tokens:
        return []
    x_min = min(t.box.x1 for t in tokens)
    x_max = max(t.box.x2 for t in tokens)
    table_width = max(1.0, x_max - x_min)

    # Token quá rộng (tiêu đề, ô merge ngang) làm nhoè biên cột nên bị loại ra khi
    # dựng band, nhưng vẫn được gán vào cột ở bước sau.
    narrow = [t for t in tokens if t.box.width <= table_width * WIDE_TOKEN_RATIO]
    source = narrow or list(tokens)

    if min_gap is None:
        heights = sorted(t.box.height for t in source)
        median_height = heights[len(heights) // 2] if heights else 10.0
        # Khoảng trắng giữa 2 cột thường lớn hơn khoảng cách giữa 2 từ cùng dòng.
        min_gap = max(6.0, median_height * 0.9)

    intervals = [(t.box.x1, t.box.x2) for t in source]
    return _merge_intervals(intervals, min_gap=min_gap)


def cluster_tokens_to_table(
    tokens: Sequence[OCRToken],
    *,
    line_overlap_threshold: float = 0.4,
    max_columns: int = 24,
) -> Table | None:
    """Dựng bảng bằng cách cluster bounding box: hàng theo y, cột theo x.

    Trả ``None`` nếu không đủ dữ liệu để dựng bảng có nghĩa (dưới 2 hàng hoặc
    không tách được cột), khi đó tầng gọi sẽ fallback sang xử lý raw text.
    """
    usable = [t for t in tokens if not t.is_empty]
    if len(usable) < 2:
        logger.warning("Không đủ token để cluster thành bảng (%d token)", len(usable))
        return None

    rows = group_tokens_into_lines(usable, overlap_threshold=line_overlap_threshold)
    if len(rows) < 2:
        logger.warning("Chỉ gom được %d hàng, không dựng được bảng", len(rows))
        return None

    bands = detect_column_bands(usable)
    if not bands:
        logger.warning("Không xác định được biên cột từ bounding box")
        return None
    if len(bands) > max_columns:
        logger.warning(
            "Phát hiện %d cột (>%d) - có thể do OCR tách từ quá vụn, vẫn tiếp tục",
            len(bands),
            max_columns,
        )
        bands = bands[:max_columns]

    grouped: dict[tuple[int, int], list[OCRToken]] = {}
    for row_index, line in enumerate(rows):
        for token in line:
            col_index = _best_band(token.box, bands)
            grouped.setdefault((row_index, col_index), []).append(token)

    cells: list[TableCell] = []
    for (row_index, col_index), cell_tokens in sorted(grouped.items()):
        box = cell_tokens[0].box
        for token in cell_tokens[1:]:
            box = box.merge(token.box)
        cells.append(
            TableCell(
                row=row_index,
                col=col_index,
                box=box,
                text=join_tokens(cell_tokens),
                tokens=list(cell_tokens),
            )
        )

    if not cells:
        return None

    table = Table(
        cells=cells,
        n_rows=len(rows),
        n_cols=len(bands),
        source="cluster",
        confidence=0.5,
    )
    logger.info("Dựng bảng bằng cluster: %d hàng x %d cột", table.n_rows, table.n_cols)
    return table


def _best_band(box: BoundingBox, bands: Sequence[tuple[float, float]]) -> int:
    """Chỉ số cột có phần chồng lấn theo trục x lớn nhất với ``box``."""
    best_index = 0
    best_overlap = -1.0
    for index, (start, end) in enumerate(bands):
        overlap = min(box.x2, end) - max(box.x1, start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = index
    return best_index


def table_from_rows(rows: Sequence[Sequence[str]], source: str = "manual") -> Table:
    """Dựng ``Table`` từ lưới text thuần - tiện cho unit test và fixture."""
    cells: list[TableCell] = []
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            cells.append(TableCell(row=row_index, col=col_index, text=str(value)))
    n_cols = max((len(row) for row in rows), default=0)
    return Table(cells=cells, n_rows=len(rows), n_cols=n_cols, source=source, confidence=1.0)
