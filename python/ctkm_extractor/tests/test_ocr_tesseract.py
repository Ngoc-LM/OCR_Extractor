"""Test riêng cho provider fallback Tesseract.

Chỉ kiểm phần LOGIC CHỌN psm - phần gọi engine được thay bằng hàm giả, nên test
không phụ thuộc chất lượng nhận dạng, chỉ cần binary tesseract tồn tại để khởi
tạo provider.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cv2", reason="Cần OpenCV để ghi ảnh tạm")
pytest.importorskip("numpy", reason="Cần numpy để dựng ảnh tạm")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ctkm_extractor.ocr.base import BoundingBox, OCRToken  # noqa: E402
from ctkm_extractor.ocr.tesseract_ocr import (  # noqa: E402
    DEFAULT_PSM_CANDIDATES,
    TesseractOCRProvider,
)

pytestmark = pytest.mark.skipif(
    not TesseractOCRProvider.is_available(), reason="Cần binary tesseract"
)


def _blank_image(tmp_path) -> str:
    path = tmp_path / "blank.png"
    cv2.imwrite(str(path), np.full((200, 1200, 3), 255, dtype=np.uint8))
    return str(path)


def _tokens(count: int, confidence: float) -> list[OCRToken]:
    return [
        OCRToken(f"t{index}", BoundingBox(0, 0, 10, 10), confidence) for index in range(count)
    ]


def test_thu_het_moi_psm_va_giu_ket_qua_tot_nhat(tmp_path, monkeypatch) -> None:
    """Regression: psm tốt nhất nằm CUỐI danh sách vẫn phải được thử.

    Bản đầu dừng sớm khi psm đầu tiên đạt >= 8 token và confidence >= 0.6. Trên
    biểu mẫu BM.12 thật, psm 4 đạt ngưỡng đó ngay nhưng không đọc được bảng CTKM,
    còn psm 6 (thử sau cùng) mới là psm đọc được.
    """
    provider = TesseractOCRProvider()
    calls: list[int] = []
    counts = {4: 8, 11: 3, 6: 20}

    def fake_run(image: object, psm: int) -> list[OCRToken]:
        calls.append(psm)
        return _tokens(counts[psm], 0.9)

    monkeypatch.setattr(provider, "_run_tesseract", fake_run)
    result = provider.extract(_blank_image(tmp_path))

    assert calls == list(DEFAULT_PSM_CANDIDATES)  # không dừng sớm
    assert len(result.tokens) == 20  # giữ lượt nhiều token nhất


def test_ep_psm_cu_the_thi_chi_chay_mot_luot(tmp_path, monkeypatch) -> None:
    provider = TesseractOCRProvider(psm=6)
    calls: list[int] = []

    def fake_run(image: object, psm: int) -> list[OCRToken]:
        calls.append(psm)
        return _tokens(5, 0.8)

    monkeypatch.setattr(provider, "_run_tesseract", fake_run)
    provider.extract(_blank_image(tmp_path))

    assert calls == [6]


def test_mot_psm_loi_khong_lam_hong_ca_luot(tmp_path, monkeypatch) -> None:
    provider = TesseractOCRProvider()

    def fake_run(image: object, psm: int) -> list[OCRToken]:
        if psm == 4:
            raise RuntimeError("engine lỗi")
        return _tokens(6, 0.7)

    monkeypatch.setattr(provider, "_run_tesseract", fake_run)
    result = provider.extract(_blank_image(tmp_path))

    assert len(result.tokens) == 6
    assert any("psm 4" in warning for warning in result.warnings)
