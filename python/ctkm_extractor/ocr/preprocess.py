"""Tiền xử lý ảnh dùng chung cho mọi OCR provider.

Pipeline: ``grayscale -> deskew -> adaptive threshold``. Mục tiêu là giảm nhiễu
watermark và làm thẳng ảnh chụp nghiêng trước khi đưa vào text detector.

``opencv-python``/``numpy`` được import mềm: nếu môi trường chấm bài không cài
được, module vẫn import bình thường và ``preprocess_image`` trả ảnh gốc kèm log
warning thay vì raise.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

try:  # pragma: no cover - phụ thuộc môi trường
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    _CV_AVAILABLE = True
except ImportError:  # pragma: no cover - phụ thuộc môi trường
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _CV_AVAILABLE = False


def is_available() -> bool:
    """True nếu OpenCV + NumPy sẵn sàng để tiền xử lý."""
    return _CV_AVAILABLE


@dataclass(frozen=True)
class PreprocessConfig:
    """Tham số tiền xử lý ảnh."""

    grayscale: bool = True
    deskew: bool = True
    adaptive_threshold: bool = True
    denoise: bool = True
    #: Góc lệch tối đa (độ) còn được coi là hợp lệ để xoay; lớn hơn thì bỏ qua.
    max_skew_angle: float = 15.0
    #: Kích thước cửa sổ của adaptive threshold (phải lẻ).
    block_size: int = 31
    #: Hằng số C trừ đi khỏi giá trị trung bình cục bộ.
    threshold_c: int = 15
    #: Phóng to ảnh nhỏ để detector bắt được chữ nhỏ (0 = tắt).
    min_width: int = 1000


DEFAULT_CONFIG = PreprocessConfig()


def load_image(image_path: str) -> Any:
    """Đọc ảnh từ đĩa thành mảng BGR. Raise ``FileNotFoundError`` nếu không đọc được."""
    if not _CV_AVAILABLE:  # pragma: no cover - phụ thuộc môi trường
        raise RuntimeError("opencv-python chưa được cài, không đọc được ảnh")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh (định dạng không hỗ trợ?): {image_path}")
    return image


def to_grayscale(image: Any) -> Any:
    """Chuyển ảnh sang thang xám (idempotent với ảnh đã 1 kênh)."""
    if not _CV_AVAILABLE:  # pragma: no cover
        return image
    if len(getattr(image, "shape", ())) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def estimate_skew_angle(gray: Any, max_angle: float = 15.0) -> float:
    """Ước lượng góc nghiêng của văn bản (độ, dương = xoay ngược chiều kim đồng hồ).

    Dùng ``minAreaRect`` trên tập pixel foreground - đủ tốt cho ảnh bảng biểu và
    rẻ hơn Hough transform.
    """
    if not _CV_AVAILABLE:  # pragma: no cover
        return 0.0
    try:
        inverted = cv2.bitwise_not(gray)
        _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        coords = cv2.findNonZero(binary)
        if coords is None or len(coords) < 50:
            return 0.0
        angle = cv2.minAreaRect(coords)[-1]
        # Quy góc về khoảng (-45, 45]. Cần CẢ HAI nhánh vì các bản OpenCV trả góc
        # theo convention khác nhau: đo trên cùng một ảnh nghiêng 2 độ, OpenCV 4.6
        # trả 2.02 còn OpenCV 5.0 trả -87.98. Thiếu nhánh cộng 90 thì trên OpenCV 5
        # mọi ảnh nghiêng nhẹ đều bị coi là lệch gần 90 độ và bỏ qua deskew.
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90
        if abs(angle) > max_angle:
            logger.debug("Bỏ qua deskew: góc ước lượng %.2f độ vượt ngưỡng", angle)
            return 0.0
        return float(angle)
    except Exception as exc:  # pragma: no cover - phòng thủ
        logger.warning("Ước lượng góc nghiêng thất bại: %s", exc)
        return 0.0


def deskew(image: Any, max_angle: float = 15.0) -> Any:
    """Xoay ảnh về phương ngang dựa trên góc nghiêng ước lượng."""
    if not _CV_AVAILABLE:  # pragma: no cover
        return image
    gray = to_grayscale(image)
    angle = estimate_skew_angle(gray, max_angle=max_angle)
    if abs(angle) < 0.15:
        return image
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    border = cv2.BORDER_REPLICATE
    rotated = cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=border
    )
    logger.debug("Đã deskew ảnh %.2f độ", angle)
    return rotated


def adaptive_threshold(gray: Any, block_size: int = 31, c: int = 15) -> Any:
    """Nhị phân hoá thích nghi - loại watermark nhạt, giữ nét chữ đậm."""
    if not _CV_AVAILABLE:  # pragma: no cover
        return gray
    block = block_size if block_size % 2 == 1 else block_size + 1
    block = max(3, block)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        c,
    )


def upscale_if_small(image: Any, min_width: int) -> Any:
    """Phóng to ảnh hẹp để detector bắt được dòng chữ nhỏ."""
    if not _CV_AVAILABLE or min_width <= 0:  # pragma: no cover
        return image
    height, width = image.shape[:2]
    if width >= min_width:
        return image
    scale = min_width / float(width)
    return cv2.resize(
        image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC
    )


def preprocess_image(
    image_path: str, config: PreprocessConfig | None = None
) -> tuple[Any, bool]:
    """Chạy toàn bộ pipeline tiền xử lý.

    Trả về ``(image, processed)``: ``image`` là mảng BGR 3 kênh (kể cả khi đã nhị
    phân hoá, để tương thích với model detect), ``processed`` cho biết pipeline có
    thực sự chạy hay chỉ trả ảnh gốc.
    """
    cfg = config or DEFAULT_CONFIG
    if not _CV_AVAILABLE:  # pragma: no cover - phụ thuộc môi trường
        logger.warning("Bỏ qua tiền xử lý: opencv-python/numpy chưa được cài")
        return image_path, False

    image = load_image(image_path)
    try:
        image = upscale_if_small(image, cfg.min_width)
        if cfg.deskew:
            image = deskew(image, max_angle=cfg.max_skew_angle)
        working = to_grayscale(image) if cfg.grayscale else image
        if cfg.grayscale and cfg.denoise:
            working = cv2.bilateralFilter(working, 5, 50, 50)
        if cfg.grayscale and cfg.adaptive_threshold:
            working = adaptive_threshold(working, cfg.block_size, cfg.threshold_c)
        if len(working.shape) == 2:
            working = cv2.cvtColor(working, cv2.COLOR_GRAY2BGR)
        return working, True
    except Exception as exc:  # pragma: no cover - phòng thủ
        logger.warning("Tiền xử lý thất bại (%s), dùng ảnh gốc", exc)
        return image, False


def save_debug_image(image: Any, out_path: str) -> str | None:
    """Ghi ảnh đã tiền xử lý ra đĩa để soi khi ``--debug``."""
    if not _CV_AVAILABLE:  # pragma: no cover
        return None
    try:
        cv2.imwrite(out_path, image)
        return out_path
    except Exception as exc:  # pragma: no cover - phòng thủ
        logger.warning("Không ghi được ảnh debug %s: %s", out_path, exc)
        return None
