"""Tách file PDF nhiều trang thành từng trang ảnh để đưa vào pipeline OCR.

``pymupdf`` được import mềm giống ``paddleocr``/``vietocr``: thiếu thì module vẫn
import được và :func:`render_pages` raise :class:`PdfUnavailableError` để tầng
trên báo lỗi tử tế thay vì traceback.

Render ở **300 DPI** - cùng độ phân giải đã dùng để kiểm chứng toàn bộ pipeline
trên biểu mẫu BM.12 thật. Thấp hơn thì tầng morphology dò thiếu đường kẻ bảng.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Sequence

logger = logging.getLogger(__name__)

#: Độ phân giải render mặc định (DPI).
DEFAULT_DPI = 300


class PdfUnavailableError(RuntimeError):
    """Không đọc được PDF (thiếu thư viện hoặc file hỏng)."""


def is_available() -> bool:
    """True nếu môi trường đọc được PDF."""
    try:
        import fitz  # type: ignore # noqa: F401
    except ImportError:
        return False
    return True


def looks_like_pdf(path: str) -> bool:
    """Đoán nhanh một đường dẫn có phải PDF không (theo đuôi file)."""
    return os.path.splitext(path)[1].lower() == ".pdf"


def parse_page_selection(spec: str | None, page_count: int) -> list[int]:
    """Diễn giải chuỗi chọn trang kiểu ``"1,3-5"`` thành danh sách chỉ số 0-based.

    Trang được đánh số **từ 1** ở giao diện người dùng. Giá trị ngoài phạm vi bị
    bỏ qua kèm cảnh báo thay vì raise - người dùng gõ nhầm không nên làm hỏng cả
    lượt chạy.
    """
    if not spec or not spec.strip():
        return list(range(page_count))

    selected: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            if "-" in chunk:
                start_text, end_text = chunk.split("-", 1)
                start, end = int(start_text), int(end_text)
            else:
                start = end = int(chunk)
        except ValueError:
            logger.warning("Bỏ qua phần chọn trang không hợp lệ: %r", chunk)
            continue
        if start > end:
            start, end = end, start
        for number in range(start, end + 1):
            index = number - 1
            if 0 <= index < page_count:
                if index not in selected:
                    selected.append(index)
            else:
                logger.warning("Bỏ qua trang %d: ngoài phạm vi 1-%d", number, page_count)
    if not selected:
        logger.warning("Không chọn được trang hợp lệ nào, dùng toàn bộ %d trang", page_count)
        return list(range(page_count))
    return selected


def page_count(pdf_path: str) -> int:
    """Số trang của file PDF."""
    fitz = _import_fitz()
    try:
        with fitz.open(pdf_path) as document:
            return int(document.page_count)
    except Exception as exc:
        raise PdfUnavailableError(f"Không đọc được PDF {pdf_path}: {exc}") from exc


def render_pages(
    pdf_path: str,
    output_dir: str,
    *,
    dpi: int = DEFAULT_DPI,
    pages: Sequence[int] | None = None,
) -> list[tuple[int, str]]:
    """Render từng trang PDF ra file PNG.

    :param pages: danh sách chỉ số trang **0-based**; ``None`` = mọi trang.
    :return: danh sách ``(số trang 1-based, đường dẫn PNG)`` theo đúng thứ tự.
    :raises PdfUnavailableError: thiếu ``pymupdf`` hoặc file không mở được.
    """
    fitz = _import_fitz()
    if not os.path.isfile(pdf_path):
        raise PdfUnavailableError(f"Không tìm thấy file PDF: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)
    rendered: list[tuple[int, str]] = []
    try:
        with fitz.open(pdf_path) as document:
            indices = list(range(document.page_count)) if pages is None else list(pages)
            for index in indices:
                if not 0 <= index < document.page_count:
                    logger.warning("Bỏ qua trang %d: ngoài phạm vi", index + 1)
                    continue
                target = os.path.join(output_dir, f"page_{index + 1:03d}.png")
                document[index].get_pixmap(dpi=dpi).save(target)
                rendered.append((index + 1, target))
                logger.debug("Đã render trang %d -> %s", index + 1, target)
    except PdfUnavailableError:
        raise
    except Exception as exc:
        raise PdfUnavailableError(f"Render PDF thất bại ({pdf_path}): {exc}") from exc

    if not rendered:
        raise PdfUnavailableError(f"PDF không có trang nào để xử lý: {pdf_path}")
    logger.info("Đã tách %d trang từ %s ở %d DPI", len(rendered), pdf_path, dpi)
    return rendered


def make_page_dir(prefix: str = "ctkm_pages_") -> str:
    """Thư mục tạm để chứa ảnh từng trang."""
    return tempfile.mkdtemp(prefix=prefix)


def _import_fitz():
    """Import ``pymupdf`` với thông báo lỗi hướng dẫn cài đặt."""
    try:
        import fitz  # type: ignore
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
        raise PdfUnavailableError(
            "Cần pymupdf để đọc PDF. Cài bằng: pip install pymupdf"
        ) from exc
    return fitz
