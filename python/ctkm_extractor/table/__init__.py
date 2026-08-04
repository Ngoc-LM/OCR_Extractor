"""Tầng dựng cấu trúc bảng, với 4 mức fallback.

1. **Morphology** (ưu tiên nhất): dò đường kẻ bảng bằng CV cổ điển - tất
   định, không cần tải model, không phụ thuộc ``paddleocr`` nên chạy được
   cả khi dùng OCR provider dự phòng (Tesseract). Phù hợp nhất với CTKM vì
   bảng có đường kẻ rõ.
2. **PP-Structure**: nhận diện hàng/cột/ô gộp bằng model học sâu - dự phòng
   cho bảng không viền hoặc layout bất thường mà (1) không dò được đường kẻ.
3. **Cluster bounding box** thủ công: gom theo y rồi theo x.
4. **Raw text blob**: không dựng bảng, để tầng trích xuất dùng regex.

Nhờ vậy pipeline không bao giờ crash khi OCR hoặc table detection sai.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..ocr.base import OCRResult
from .morphology import (
    MorphologyTableRecognizer,
    TableMorphologyUnavailableError,
)
from .pp_structure import (
    PPStructureTableRecognizer,
    TableStructureUnavailableError,
)
from .reconstruct import (
    Table,
    TableCell,
    assign_tokens_to_cells,
    cluster_tokens_to_table,
    join_tokens,
    table_from_rows,
)

logger = logging.getLogger(__name__)

STRATEGY_MORPHOLOGY = "morphology"
STRATEGY_PP_STRUCTURE = "pp_structure"
STRATEGY_CLUSTER = "cluster"
STRATEGY_RAW_TEXT = "raw_text"


@dataclass
class TableBuildResult:
    """Kết quả dựng bảng kèm chiến lược đã dùng và cảnh báo phát sinh."""

    table: Table | None
    strategy: str
    raw_text: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def has_table(self) -> bool:
        return self.table is not None and not self.table.is_empty


def build_table(
    ocr_result: OCRResult,
    *,
    image: Any = None,
    use_morphology: bool = True,
    use_pp_structure: bool = True,
    morphology_recognizer: MorphologyTableRecognizer | None = None,
    recognizer: PPStructureTableRecognizer | None = None,
) -> TableBuildResult:
    """Dựng bảng từ kết quả OCR, tự động hạ cấp chiến lược khi thất bại.

    :param ocr_result: token + toạ độ do tầng OCR trả về.
    :param image: ảnh (đường dẫn hoặc mảng) cho morphology/PP-Structure;
        ``None`` thì dùng ``ocr_result.image_path``.
    :param use_morphology: đặt False để bỏ qua mức 1 (hữu ích khi test).
    :param use_pp_structure: đặt False để bỏ qua mức 2 (hữu ích khi test).
    :param morphology_recognizer: cho phép inject recognizer đã khởi tạo
        sẵn / mock trong test.
    :param recognizer: cho phép inject PP-Structure recognizer đã khởi tạo
        sẵn / mock trong test.
    """
    warnings: list[str] = []
    tokens = ocr_result.non_empty_tokens()
    raw_text = ocr_result.raw_text

    if not tokens:
        message = "OCR không trả về token nào - chuyển sang xử lý raw text"
        logger.warning(message)
        warnings.append(message)
        return TableBuildResult(None, STRATEGY_RAW_TEXT, raw_text, warnings)

    # Ưu tiên ảnh ĐÃ TIỀN XỬ LÝ mà provider dùng để sinh token: bounding box của
    # token nằm trong hệ toạ độ ảnh đó. Đọc lại ảnh gốc từ đĩa sẽ lệch hệ toạ độ
    # khi tiền xử lý có phóng to/xoay ảnh - và với ảnh scan bị nghiêng thì không
    # dò được đường kẻ ngang nào nên morphology bị vô hiệu.
    if image is not None:
        target_image = image
    elif ocr_result.processed_image is not None:
        target_image = ocr_result.processed_image
    else:
        target_image = ocr_result.image_path

    if use_morphology and target_image is not None:
        try:
            engine = morphology_recognizer or MorphologyTableRecognizer()
            if engine.is_available():
                table = engine.recognize(target_image, tokens)
                return TableBuildResult(table, STRATEGY_MORPHOLOGY, raw_text, warnings)
            warnings.append("Thiếu OpenCV cho morphology - fallback PP-Structure")
        except TableMorphologyUnavailableError as exc:
            message = f"Morphology không dựng được bảng ({exc}) - fallback PP-Structure"
            logger.info(message)
            warnings.append(message)
        except Exception as exc:  # pragma: no cover - phòng thủ
            message = f"Morphology lỗi bất ngờ ({exc}) - fallback PP-Structure"
            logger.warning(message)
            warnings.append(message)
    elif use_morphology:
        warnings.append("Không có ảnh nguồn cho morphology - fallback PP-Structure")

    if use_pp_structure and target_image is not None:
        try:
            engine = recognizer or PPStructureTableRecognizer()
            table = engine.recognize(target_image, tokens)
            return TableBuildResult(table, STRATEGY_PP_STRUCTURE, raw_text, warnings)
        except TableStructureUnavailableError as exc:
            message = f"PP-Structure không dùng được ({exc}) - fallback cluster bounding box"
            logger.warning(message)
            warnings.append(message)
        except Exception as exc:  # pragma: no cover - phòng thủ
            message = f"PP-Structure lỗi bất ngờ ({exc}) - fallback cluster bounding box"
            logger.warning(message)
            warnings.append(message)
    elif use_pp_structure:
        warnings.append("Không có ảnh nguồn cho PP-Structure - fallback cluster bounding box")

    try:
        table = cluster_tokens_to_table(tokens)
    except Exception as exc:  # pragma: no cover - phòng thủ
        logger.warning("Cluster bounding box lỗi: %s", exc)
        warnings.append(f"Cluster bounding box lỗi: {exc}")
        table = None

    if table is not None and not table.is_empty:
        return TableBuildResult(table, STRATEGY_CLUSTER, raw_text, warnings)

    message = "Không dựng được bảng - xử lý toàn bộ text như một khối duy nhất"
    logger.warning(message)
    warnings.append(message)
    return TableBuildResult(None, STRATEGY_RAW_TEXT, raw_text, warnings)


__all__ = [
    "MorphologyTableRecognizer",
    "PPStructureTableRecognizer",
    "STRATEGY_CLUSTER",
    "STRATEGY_MORPHOLOGY",
    "STRATEGY_PP_STRUCTURE",
    "STRATEGY_RAW_TEXT",
    "Table",
    "TableBuildResult",
    "TableCell",
    "TableMorphologyUnavailableError",
    "TableStructureUnavailableError",
    "assign_tokens_to_cells",
    "build_table",
    "cluster_tokens_to_table",
    "join_tokens",
    "table_from_rows",
]
