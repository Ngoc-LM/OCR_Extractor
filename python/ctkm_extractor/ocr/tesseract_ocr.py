"""Provider fallback nhẹ: Tesseract OCR qua ``pytesseract``.

Dùng khi môi trường không cài được ``paddlepaddle``/``paddleocr``/``vietocr``
(model nặng, cần tải weights lần đầu). Đánh đổi: reproducibility cao (chỉ cần
``apt-get install tesseract-ocr tesseract-ocr-vie``) nhưng độ chính xác dấu tiếng
Việt thấp hơn rõ rệt so với provider mặc định.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import (
    BoundingBox,
    OCRProvider,
    OCRResult,
    OCRToken,
    ProviderUnavailableError,
)
from .preprocess import PreprocessConfig, is_available as cv_available, preprocess_image

logger = logging.getLogger(__name__)

#: Thứ tự thử ngôn ngữ: ưu tiên gói tiếng Việt, sau đó mới tới tiếng Anh.
DEFAULT_LANGS = ("vie", "vie+eng", "eng")

#: Các chế độ phân đoạn trang thử lần lượt khi người dùng không chỉ định ``--psm``.
#: 4 = một cột văn bản nhiều cỡ chữ, 11 = text rời rạc, 6 = một khối đồng nhất.
#: Bảng có đường kẻ thường bị psm 6 (mặc định của Tesseract) cắt sai dòng.
DEFAULT_PSM_CANDIDATES = (4, 11, 6)

#: LƯU Ý: KHÔNG dừng sớm ở psm đầu tiên "trông có vẻ ổn".
#:
#: Bản đầu có luật dừng sớm (>= 8 token và confidence >= 0.6 thì bỏ qua các psm
#: còn lại). Trên biểu mẫu BM.12 thật (render 300 DPI), psm 4 được thử đầu tiên
#: và đạt ngay ngưỡng đó (80 token, confidence 0.82) nên vòng lặp dừng lại - dù
#: nó KHÔNG đọc được bảng CTKM bên trong. Đo trên cùng ảnh: psm 4 = 79 token,
#: psm 11 = 161 token, psm 6 = 195 token và chỉ psm 6 đọc ra được "CTKMN180X",
#: "163,636.3636", "150.534.213". Vì vậy luôn chạy hết mọi psm rồi mới chọn.


class TesseractOCRProvider(OCRProvider):
    """Nhận dạng bằng Tesseract, lấy cả bounding box từng từ qua ``image_to_data``."""

    name = "tesseract"

    def __init__(
        self,
        *,
        lang: str | None = None,
        psm: int | None = None,
        oem: int = 3,
        min_confidence: float = 0.0,
        preprocess_config: PreprocessConfig | None = None,
    ) -> None:
        """Khởi tạo provider; raise ``ProviderUnavailableError`` nếu thiếu binary.

        ``psm=None`` (mặc định) sẽ thử lần lượt :data:`DEFAULT_PSM_CANDIDATES` và giữ
        kết quả tốt nhất; truyền số cụ thể để ép một chế độ duy nhất.
        """
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
        except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
            raise ProviderUnavailableError(
                f"Thiếu pytesseract/Pillow cho provider tesseract: {exc}"
            ) from exc

        self._pytesseract = pytesseract
        self._pil_image = Image
        self.psm = psm
        self.oem = oem
        self.min_confidence = min_confidence
        self.preprocess_config = preprocess_config or PreprocessConfig()

        try:
            available = set(pytesseract.get_languages(config=""))
        except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
            raise ProviderUnavailableError(
                f"Không gọi được binary tesseract (đã cài chưa?): {exc}"
            ) from exc

        self.lang = self._pick_language(lang, available)

    @staticmethod
    def _pick_language(requested: str | None, available: set[str]) -> str:
        """Chọn gói ngôn ngữ tốt nhất đang có; cảnh báo nếu thiếu gói tiếng Việt."""
        if requested:
            missing = [part for part in requested.split("+") if part not in available]
            if missing:
                logger.warning(
                    "Tesseract thiếu gói ngôn ngữ %s, thử dùng mặc định", ", ".join(missing)
                )
            else:
                return requested
        for candidate in DEFAULT_LANGS:
            if all(part in available for part in candidate.split("+")):
                if candidate == "eng":
                    logger.warning(
                        "Không tìm thấy gói 'vie' - độ chính xác dấu tiếng Việt sẽ rất thấp. "
                        "Cài bằng: apt-get install tesseract-ocr-vie"
                    )
                return candidate
        logger.warning("Không xác định được gói ngôn ngữ Tesseract, dùng 'eng'")
        return "eng"

    @classmethod
    def is_available(cls) -> bool:
        """True nếu ``pytesseract`` import được và binary tesseract chạy được."""
        try:
            import pytesseract  # type: ignore
        except ImportError:
            return False
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            return False
        return True

    def _load_image(self, image_path: str) -> tuple[Any, Any, list[str]]:
        """Đọc ảnh (đã tiền xử lý nếu OpenCV có sẵn) thành đối tượng PIL.

        Trả ``(ảnh PIL, mảng BGR đã tiền xử lý hoặc None, cảnh báo)``. Mảng BGR
        được trả kèm để tầng dựng bảng dò đường kẻ trên ĐÚNG ảnh đã sinh ra token.
        """
        warnings: list[str] = []
        if cv_available():
            processed, ok = preprocess_image(image_path, self.preprocess_config)
            if ok and not isinstance(processed, str):
                try:
                    import cv2  # type: ignore

                    rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                    return self._pil_image.fromarray(rgb), processed, warnings
                except Exception as exc:  # pragma: no cover - phòng thủ
                    warnings.append(f"Không chuyển được ảnh tiền xử lý sang PIL: {exc}")
            else:
                warnings.append("Không tiền xử lý được ảnh, dùng ảnh gốc")
        else:
            warnings.append("Thiếu opencv-python, bỏ qua tiền xử lý")
        try:
            return self._pil_image.open(image_path), None, warnings
        except Exception as exc:
            raise ProviderUnavailableError(f"Không đọc được ảnh {image_path}: {exc}") from exc

    def extract(self, image_path: str) -> OCRResult:
        """Chạy Tesseract và chuyển ``image_to_data`` thành danh sách token.

        Khi không ép ``psm``, chạy HẾT các chế độ phân đoạn rồi giữ kết quả có
        nhiều token nhất (tie-break bằng confidence trung bình) - xem ghi chú ở
        :data:`DEFAULT_PSM_CANDIDATES` về việc vì sao không được dừng sớm.
        """
        image, processed_image, warnings = self._load_image(image_path)
        candidates = [self.psm] if self.psm is not None else list(DEFAULT_PSM_CANDIDATES)

        best_tokens: list[OCRToken] = []
        best_key: tuple[int, float] = (-1, -1.0)
        best_psm: int | None = None
        errors: list[str] = []

        for psm in candidates:
            try:
                tokens = self._run_tesseract(image, psm)
            except Exception as exc:
                logger.warning("Tesseract lỗi với --psm %s trên %s: %s", psm, image_path, exc)
                errors.append(f"psm {psm}: {exc}")
                continue
            confidence = (
                sum(token.confidence for token in tokens) / len(tokens) if tokens else 0.0
            )
            key = (len(tokens), confidence)
            if key > best_key:
                best_key, best_tokens, best_psm = key, tokens, psm

        if best_psm is None:
            return OCRResult(
                tokens=[],
                image_path=image_path,
                provider=self.name,
                warnings=warnings + errors,
                processed_image=processed_image,
            )

        logger.debug(
            "Tesseract chọn --psm %s (%d token, confidence %.2f)",
            best_psm,
            best_key[0],
            best_key[1],
        )
        if not best_tokens:
            warnings.append("Tesseract không nhận dạng được text nào")
            logger.warning("Tesseract không nhận dạng được text nào trong %s", image_path)

        size = getattr(image, "size", None)
        return OCRResult(
            tokens=best_tokens,
            image_path=image_path,
            image_size=(int(size[0]), int(size[1])) if size else None,
            provider=self.name,
            warnings=warnings + errors,
            processed_image=processed_image,
        )

    def _run_tesseract(self, image: Any, psm: int) -> list[OCRToken]:
        """Chạy một lượt ``image_to_data`` với chế độ phân đoạn ``psm``."""
        data = self._pytesseract.image_to_data(
            image,
            lang=self.lang,
            config=f"--oem {self.oem} --psm {psm}",
            output_type=self._pytesseract.Output.DICT,
        )

        tokens: list[OCRToken] = []
        count = len(data.get("text", []))
        for index in range(count):
            text = (data["text"][index] or "").strip()
            if not text:
                continue
            try:
                confidence = float(data["conf"][index])
            except (TypeError, ValueError):
                confidence = -1.0
            # Tesseract dùng -1 cho block/paragraph, không phải token thật.
            if confidence < 0:
                continue
            confidence /= 100.0
            if confidence < self.min_confidence:
                continue
            try:
                box = BoundingBox.from_xywh(
                    float(data["left"][index]),
                    float(data["top"][index]),
                    float(data["width"][index]),
                    float(data["height"][index]),
                )
            except (TypeError, ValueError, KeyError):
                continue
            tokens.append(OCRToken(text=text, box=box, confidence=confidence))

        return tokens


#: Alias ngắn gọn theo tên đặt trong đề bài.
TesseractOCR = TesseractOCRProvider
