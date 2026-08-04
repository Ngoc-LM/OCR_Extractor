"""Kiểu dữ liệu và interface chung cho tầng OCR.

Mọi provider (PaddleOCR-detector + VietOCR, Tesseract, ...) đều trả về cùng một
cấu trúc :class:`OCRResult` gồm danh sách token ``text + bounding box +
confidence``. Nhờ vậy các tầng phía sau (table, extraction) hoàn toàn không phụ
thuộc vào engine OCR cụ thể.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterable, Sequence

logger = logging.getLogger(__name__)

Point = tuple[float, float]
Polygon = tuple[Point, ...]


class OCRError(RuntimeError):
    """Lỗi chung của tầng OCR."""


class ProviderUnavailableError(OCRError):
    """Provider không dùng được (thiếu dependency, thiếu binary, thiếu model)."""


@dataclass(frozen=True)
class BoundingBox:
    """Hộp bao (axis-aligned) của một token text, đơn vị pixel."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        # Chuẩn hoá để x1 <= x2 và y1 <= y2, tránh box "âm" do OCR trả ngược toạ độ.
        x1, x2 = float(self.x1), float(self.x2)
        y1, y2 = float(self.y1), float(self.y2)
        object.__setattr__(self, "x1", min(x1, x2))
        object.__setattr__(self, "x2", max(x1, x2))
        object.__setattr__(self, "y1", min(y1, y2))
        object.__setattr__(self, "y2", max(y1, y2))

    @classmethod
    def from_polygon(cls, points: Iterable[Sequence[float]]) -> "BoundingBox":
        """Tạo box từ polygon 4 điểm (hoặc nhiều hơn) do detector trả về."""
        xs: list[float] = []
        ys: list[float] = []
        for point in points:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
        if not xs or not ys:
            raise ValueError("polygon rỗng, không dựng được bounding box")
        return cls(min(xs), min(ys), max(xs), max(ys))

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "BoundingBox":
        """Tạo box từ định dạng ``(x, y, width, height)`` (Tesseract dùng dạng này)."""
        return cls(x, y, x + w, y + h)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Point:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def intersection_area(self, other: "BoundingBox") -> float:
        """Diện tích giao nhau giữa hai box (0 nếu rời nhau)."""
        dx = min(self.x2, other.x2) - max(self.x1, other.x1)
        dy = min(self.y2, other.y2) - max(self.y1, other.y1)
        if dx <= 0 or dy <= 0:
            return 0.0
        return dx * dy

    def overlap_ratio(self, other: "BoundingBox") -> float:
        """Tỉ lệ diện tích của ``self`` nằm trong ``other`` (0..1)."""
        if self.area <= 0:
            return 0.0
        return self.intersection_area(other) / self.area

    def iou(self, other: "BoundingBox") -> float:
        """Intersection-over-Union, dùng khi so khớp ô bảng với token."""
        inter = self.intersection_area(other)
        union = self.area + other.area - inter
        if union <= 0:
            return 0.0
        return inter / union

    def contains_center_of(self, other: "BoundingBox") -> bool:
        """True nếu tâm của ``other`` nằm trong ``self``."""
        cx, cy = other.center
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2

    def vertical_overlap_ratio(self, other: "BoundingBox") -> float:
        """Tỉ lệ chồng lấn theo trục y so với chiều cao nhỏ hơn (dùng gom dòng)."""
        dy = min(self.y2, other.y2) - max(self.y1, other.y1)
        if dy <= 0:
            return 0.0
        base = min(self.height, other.height)
        if base <= 0:
            return 0.0
        return dy / base

    def merge(self, other: "BoundingBox") -> "BoundingBox":
        """Hộp bao nhỏ nhất chứa cả hai box."""
        return BoundingBox(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            max(self.x2, other.x2),
            max(self.y2, other.y2),
        )


@dataclass
class OCRToken:
    """Một đoạn text đã nhận dạng kèm vị trí và độ tin cậy."""

    text: str
    box: BoundingBox
    confidence: float = 1.0
    polygon: Polygon | None = None

    def __post_init__(self) -> None:
        self.text = (self.text or "").strip()
        try:
            self.confidence = float(self.confidence)
        except (TypeError, ValueError):
            self.confidence = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.text


@dataclass
class OCRResult:
    """Kết quả OCR của một ảnh."""

    tokens: list[OCRToken] = field(default_factory=list)
    image_path: str | None = None
    image_size: tuple[int, int] | None = None  # (width, height)
    provider: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    #: Ảnh ĐÃ TIỀN XỬ LÝ mà provider dùng để sinh token (mảng BGR), nếu có.
    #:
    #: Bắt buộc phải mang theo vì bounding box của token nằm trong hệ toạ độ của
    #: ảnh này: tiền xử lý có thể phóng to (``min_width``) và xoay (deskew) ảnh.
    #: Tầng dựng bảng phải dò đường kẻ trên đúng ảnh đó, đọc lại ảnh gốc từ đĩa sẽ
    #: lệch hệ toạ độ - và trên ảnh scan bị nghiêng thì không dò được đường kẻ
    #: ngang nào, khiến morphology (mức dựng bảng tốt nhất) bị vô hiệu.
    processed_image: Any = None

    def non_empty_tokens(self) -> list[OCRToken]:
        return [t for t in self.tokens if not t.is_empty]

    @property
    def raw_text(self) -> str:
        """Toàn bộ text gom theo dòng (trên xuống dưới, trái sang phải)."""
        lines = group_tokens_into_lines(self.non_empty_tokens())
        return "\n".join(" ".join(tok.text for tok in line) for line in lines)

    @property
    def mean_confidence(self) -> float:
        tokens = self.non_empty_tokens()
        if not tokens:
            return 0.0
        return sum(t.confidence for t in tokens) / len(tokens)

    @classmethod
    def from_text(cls, text: str, provider: str = "text") -> "OCRResult":
        """Dựng ``OCRResult`` giả lập từ text thuần - hữu ích cho test và debug.

        Mỗi dòng thành một token với bounding box tổng hợp theo chỉ số dòng, đủ để
        tầng dựng bảng và tầng trích xuất hoạt động mà không cần ảnh thật.
        """
        tokens: list[OCRToken] = []
        line_height = 20.0
        char_width = 10.0
        for row, line in enumerate(text.splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            y1 = row * (line_height + 6.0)
            width = max(1, len(stripped)) * char_width
            tokens.append(
                OCRToken(
                    text=stripped,
                    box=BoundingBox(0.0, y1, width, y1 + line_height),
                    confidence=1.0,
                )
            )
        return cls(tokens=tokens, provider=provider)


def sort_tokens_reading_order(tokens: Sequence[OCRToken]) -> list[OCRToken]:
    """Sắp xếp token theo thứ tự đọc (y trước, x sau)."""
    return sorted(tokens, key=lambda t: (t.box.y1, t.box.x1))


def group_tokens_into_lines(
    tokens: Sequence[OCRToken], overlap_threshold: float = 0.4
) -> list[list[OCRToken]]:
    """Gom token thành các dòng dựa trên độ chồng lấn theo trục y.

    Dùng chung cho ``OCRResult.raw_text`` và tầng dựng bảng fallback.
    """
    lines: list[list[OCRToken]] = []
    for token in sort_tokens_reading_order(tokens):
        placed = False
        for line in lines:
            # So với token cuối cùng của dòng: đủ nhanh và ổn định với bảng.
            reference = line[-1]
            if reference.box.vertical_overlap_ratio(token.box) >= overlap_threshold:
                line.append(token)
                placed = True
                break
        if not placed:
            lines.append([token])
    for line in lines:
        line.sort(key=lambda t: t.box.x1)
    lines.sort(key=lambda line: min(t.box.y1 for t in line))
    return lines


class OCRProvider(ABC):
    """Interface mà mọi engine OCR phải hiện thực."""

    #: Tên ngắn dùng cho CLI (``--engine``) và log.
    name: ClassVar[str] = "base"

    @classmethod
    def is_available(cls) -> bool:
        """True nếu provider chạy được trong môi trường hiện tại."""
        return False

    @abstractmethod
    def extract(self, image_path: str) -> OCRResult:
        """Chạy OCR trên ảnh và trả về danh sách token có toạ độ."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - tiện debug
        return f"<{type(self).__name__} name={self.name!r}>"
