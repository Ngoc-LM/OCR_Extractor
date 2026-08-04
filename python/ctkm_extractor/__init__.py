"""CTKM Extractor.

Pipeline OCR + structured extraction: đọc ảnh bảng mô tả chương trình khuyến mại
(CTKM) viễn thông và trích xuất dữ liệu ra JSON.

Kiến trúc 4 tầng, mỗi tầng test được độc lập:

* ``ctkm_extractor.ocr``        - detection (PP-OCRv5 / DB) + recognition (VietOCR),
  fallback Tesseract.
* ``ctkm_extractor.table``      - dựng cấu trúc bảng (PP-Structure -> cluster thủ công
  -> raw text blob).
* ``ctkm_extractor.extraction`` - map field theo ``schema.yaml`` + parser registry.
* ``ctkm_extractor.cli``        - giao diện dòng lệnh.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
