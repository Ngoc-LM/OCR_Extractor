"""Nhận diện cấu trúc bảng bằng PP-Structure (ưu tiên số 1).

Bảng CTKM có nhiều ô merge theo chiều dọc (cột "Nội dung"), tự cluster bounding
box theo x/y rất dễ sai ở những ô này. PP-Structure trả trực tiếp cấu trúc
hàng/cột/ô gộp dưới dạng HTML + toạ độ từng ô, nên ta:

1. Lấy cấu trúc ô (vị trí lưới + span + bounding box) từ PP-Structure.
2. Ghép text đã nhận dạng bởi **VietOCR** vào từng ô theo diện tích chồng lấn.

Recognizer mặc định của PaddleOCR không được dùng (``ocr=False``), chỉ lấy phần
cấu trúc.
"""

from __future__ import annotations

import logging
from html.parser import HTMLParser
from typing import Any, Iterable, Sequence

from ..ocr.base import BoundingBox, OCRToken
from .reconstruct import Table, TableCell, assign_tokens_to_cells

logger = logging.getLogger(__name__)

#: Tỉ lệ token được gán vào ô tối thiểu để coi kết quả PP-Structure là đáng tin.
MIN_ASSIGN_RATIO = 0.5


class TableStructureUnavailableError(RuntimeError):
    """PP-Structure không dùng được hoặc không cho kết quả đáng tin cậy."""


class _HTMLTableParser(HTMLParser):
    """Phân tích HTML bảng của PP-Structure thành danh sách vị trí ô.

    Chỉ quan tâm cấu trúc (``<tr>``, ``<td>``, ``rowspan``, ``colspan``); nội dung
    text bên trong bị bỏ qua vì text sẽ lấy từ VietOCR.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cells: list[dict[str, int]] = []
        self._row = -1
        self._occupied: set[tuple[int, int]] = set()
        self._next_col = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row += 1
            self._next_col = 0
            return
        if tag not in ("td", "th"):
            return
        if self._row < 0:  # td nằm ngoài tr (HTML lỗi) -> coi như hàng 0
            self._row = 0
        attributes = {key: (value or "") for key, value in attrs}
        row_span = _to_positive_int(attributes.get("rowspan"), default=1)
        col_span = _to_positive_int(attributes.get("colspan"), default=1)

        column = self._next_col
        while (self._row, column) in self._occupied:
            column += 1

        for r in range(self._row, self._row + row_span):
            for c in range(column, column + col_span):
                self._occupied.add((r, c))

        self.cells.append(
            {
                "row": self._row,
                "col": column,
                "row_span": row_span,
                "col_span": col_span,
            }
        )
        self._next_col = column + col_span


def _to_positive_int(value: Any, default: int = 1) -> int:
    """Ép về số nguyên dương, trả ``default`` khi giá trị hỏng."""
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def parse_table_html(html: str) -> list[dict[str, int]]:
    """Trả danh sách ô ``{row, col, row_span, col_span}`` theo đúng thứ tự ``<td>``."""
    parser = _HTMLTableParser()
    try:
        parser.feed(html or "")
        parser.close()
    except Exception as exc:  # pragma: no cover - phòng thủ với HTML hỏng
        logger.warning("Không parse được HTML bảng của PP-Structure: %s", exc)
        return []
    return parser.cells


def _normalize_cell_bbox(raw: Any) -> BoundingBox | None:
    """Chuẩn hoá bbox ô: chấp nhận ``[x1,y1,x2,y2]`` hoặc polygon 4/8 giá trị."""
    if raw is None:
        return None
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    try:
        values = list(raw)
    except TypeError:
        return None
    if not values:
        return None
    if isinstance(values[0], (list, tuple)):
        try:
            return BoundingBox.from_polygon(values)
        except (ValueError, TypeError, IndexError):
            return None
    try:
        numbers = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    if len(numbers) == 4:
        return BoundingBox(numbers[0], numbers[1], numbers[2], numbers[3])
    if len(numbers) >= 8:
        points = [(numbers[i], numbers[i + 1]) for i in range(0, 8, 2)]
        try:
            return BoundingBox.from_polygon(points)
        except (ValueError, TypeError):
            return None
    return None


class PPStructureTableRecognizer:
    """Bọc ``paddleocr.PPStructure`` để lấy cấu trúc bảng (không lấy text)."""

    name = "pp_structure"

    def __init__(self, *, use_gpu: bool = False, lang: str = "en") -> None:
        """Khởi tạo model; raise ``TableStructureUnavailableError`` nếu không sẵn sàng."""
        self._engine = self._build_engine(use_gpu=use_gpu, lang=lang)

    @staticmethod
    def _build_engine(*, use_gpu: bool, lang: str) -> Any:
        """Khởi tạo engine PP-Structure, thử lần lượt API của các phiên bản khác nhau."""
        try:
            import paddleocr  # type: ignore
        except ImportError as exc:
            raise TableStructureUnavailableError(
                f"Thiếu paddleocr/paddlepaddle cho PP-Structure: {exc}"
            ) from exc

        candidates: list[tuple[str, dict[str, Any]]] = [
            ("PPStructure", {"table": True, "ocr": False, "show_log": False, "lang": lang}),
            ("PPStructure", {"table": True, "ocr": False, "lang": lang}),
            ("PPStructure", {"table": True, "ocr": False}),
            ("PPStructureV3", {"use_doc_orientation_classify": False, "use_doc_unwarping": False}),
            ("PPStructureV3", {}),
        ]
        if use_gpu:
            for _, kwargs in candidates:
                kwargs.setdefault("use_gpu", True)

        last_error: Exception | None = None
        for attribute, kwargs in candidates:
            engine_cls = getattr(paddleocr, attribute, None)
            if engine_cls is None:
                continue
            try:
                return engine_cls(**kwargs)
            except Exception as exc:  # pragma: no cover - phụ thuộc phiên bản
                last_error = exc
                logger.debug("Khởi tạo %s(%s) thất bại: %s", attribute, kwargs, exc)
        raise TableStructureUnavailableError(
            f"Không khởi tạo được PP-Structure: {last_error}"
        )

    @classmethod
    def is_available(cls) -> bool:
        """True nếu ``paddleocr`` có sẵn class PP-Structure."""
        try:
            import paddleocr  # type: ignore
        except ImportError:
            return False
        return any(hasattr(paddleocr, name) for name in ("PPStructure", "PPStructureV3"))

    def _run(self, image: Any) -> list[dict[str, Any]]:
        """Gọi engine và trả về list kết quả dạng dict."""
        raw: Any = None
        for call in (
            lambda: self._engine(image),
            lambda: self._engine.predict(image),
        ):
            try:
                raw = call()
                if raw is not None:
                    break
            except Exception as exc:  # pragma: no cover - phụ thuộc phiên bản
                logger.debug("Gọi PP-Structure thất bại: %s", exc)
        if raw is None:
            raise TableStructureUnavailableError("PP-Structure không trả về kết quả")
        return _flatten_results(raw)

    def recognize(self, image: Any, tokens: Sequence[OCRToken]) -> Table:
        """Dựng :class:`Table` từ cấu trúc PP-Structure + text VietOCR.

        :param image: đường dẫn ảnh hoặc mảng ảnh đã tiền xử lý.
        :param tokens: token đã nhận dạng bởi VietOCR ở tầng OCR.
        :raises TableStructureUnavailableError: khi không có bảng hoặc kết quả
            không đủ tin cậy (ít token khớp ô) - tầng gọi sẽ fallback.
        """
        results = self._run(image)
        best_table: Table | None = None
        best_score = 0.0

        for item in results:
            structure = _extract_table_payload(item)
            if structure is None:
                continue
            html, bboxes = structure
            cells = self._build_cells(html, bboxes)
            if not cells:
                continue
            assigned = assign_tokens_to_cells(cells, tokens)
            usable_tokens = [t for t in tokens if not t.is_empty]
            ratio = assigned / len(usable_tokens) if usable_tokens else 0.0
            if ratio > best_score:
                best_score = ratio
                best_table = Table(cells=cells, source=self.name, confidence=ratio)

        if best_table is None:
            raise TableStructureUnavailableError("PP-Structure không tìm thấy bảng nào")
        if best_score < MIN_ASSIGN_RATIO:
            raise TableStructureUnavailableError(
                f"Chỉ ghép được {best_score:.0%} token vào ô - kết quả không đáng tin"
            )
        logger.info(
            "PP-Structure dựng bảng %dx%d, ghép %.0f%% token",
            best_table.n_rows,
            best_table.n_cols,
            best_score * 100,
        )
        return best_table

    @staticmethod
    def _build_cells(html: str, bboxes: Sequence[Any]) -> list[TableCell]:
        """Ghép vị trí ô (từ HTML) với bounding box tương ứng (cùng thứ tự)."""
        positions = parse_table_html(html)
        boxes = [_normalize_cell_bbox(b) for b in bboxes]
        boxes = [b for b in boxes if b is not None]

        if positions and boxes and len(positions) == len(boxes):
            return [
                TableCell(
                    row=position["row"],
                    col=position["col"],
                    row_span=position["row_span"],
                    col_span=position["col_span"],
                    box=box,
                )
                for position, box in zip(positions, boxes)
            ]

        if boxes:
            logger.warning(
                "Số ô trong HTML (%d) khác số bbox (%d) - suy lưới từ toạ độ bbox",
                len(positions),
                len(boxes),
            )
            return _cells_from_boxes(boxes)

        logger.warning("PP-Structure không trả bbox cho ô nào")
        return []


def _cells_from_boxes(boxes: Sequence[BoundingBox]) -> list[TableCell]:
    """Suy chỉ số hàng/cột từ toạ độ ô khi HTML và bbox không khớp số lượng."""
    if not boxes:
        return []
    row_edges = _cluster_positions([b.y1 for b in boxes])
    col_edges = _cluster_positions([b.x1 for b in boxes])
    cells: list[TableCell] = []
    for box in boxes:
        row = _nearest_index(box.y1, row_edges)
        col = _nearest_index(box.x1, col_edges)
        row_end = _nearest_index(box.y2, row_edges)
        col_end = _nearest_index(box.x2, col_edges)
        cells.append(
            TableCell(
                row=row,
                col=col,
                row_span=max(1, row_end - row),
                col_span=max(1, col_end - col),
                box=box,
            )
        )
    return cells


def _cluster_positions(values: Iterable[float], tolerance: float = 12.0) -> list[float]:
    """Gom các toạ độ gần nhau thành một mốc lưới duy nhất."""
    ordered = sorted(values)
    edges: list[float] = []
    for value in ordered:
        if not edges or value - edges[-1] > tolerance:
            edges.append(value)
    return edges


def _nearest_index(value: float, edges: Sequence[float]) -> int:
    """Chỉ số mốc lưới gần ``value`` nhất."""
    if not edges:
        return 0
    best = 0
    best_distance = abs(value - edges[0])
    for index, edge in enumerate(edges[1:], start=1):
        distance = abs(value - edge)
        if distance < best_distance:
            best_distance = distance
            best = index
    return best


def _flatten_results(raw: Any) -> list[dict[str, Any]]:
    """Chuẩn hoá output PP-Structure về ``list[dict]`` cho mọi phiên bản."""
    results: list[dict[str, Any]] = []

    def walk(node: Any, depth: int = 0) -> None:
        if node is None or depth > 4:
            return
        if isinstance(node, dict):
            results.append(node)
            return
        if hasattr(node, "json"):  # pragma: no cover - phụ thuộc phiên bản
            try:
                walk(node.json, depth + 1)
                return
            except Exception:
                pass
        if isinstance(node, (list, tuple)):
            for item in node:
                walk(item, depth + 1)

    walk(raw)
    return results


def _extract_table_payload(item: dict[str, Any]) -> tuple[str, list[Any]] | None:
    """Lấy ``(html, cell_bboxes)`` từ một phần tử kết quả PP-Structure."""
    if not isinstance(item, dict):
        return None
    if item.get("type") not in (None, "table"):
        return None
    payload = item.get("res", item)
    if not isinstance(payload, dict):
        return None
    html = payload.get("html") or payload.get("pred_html") or ""
    bboxes = (
        payload.get("cell_bbox")
        or payload.get("cell_box_list")
        or payload.get("cell_bbox_list")
        or payload.get("bbox")
        or []
    )
    if not html and not bboxes:
        return None
    try:
        bbox_list = list(bboxes)
    except TypeError:
        bbox_list = []
    return str(html), bbox_list
