"""Tầng OCR: interface chung + factory chọn provider kèm fallback tự động."""

from __future__ import annotations

import logging
from typing import Any, Callable

from .base import (
    BoundingBox,
    OCRError,
    OCRProvider,
    OCRResult,
    OCRToken,
    ProviderUnavailableError,
    group_tokens_into_lines,
    sort_tokens_reading_order,
)
from .preprocess import PreprocessConfig

logger = logging.getLogger(__name__)

#: Tên engine hợp lệ cho CLI, theo thứ tự ưu tiên khi chạy ``auto``.
ENGINE_PRIORITY: tuple[str, ...] = ("paddle_vietocr", "tesseract")

AUTO_ENGINE = "auto"


def _load_paddle_vietocr(**kwargs: Any) -> OCRProvider:
    """Import muộn provider mặc định để tránh kéo torch/paddle khi không cần."""
    from .paddle_vietocr_provider import PaddleDetectorVietOCRProvider

    return PaddleDetectorVietOCRProvider(**kwargs)


def _load_tesseract(**kwargs: Any) -> OCRProvider:
    """Import muộn provider fallback."""
    from .tesseract_ocr import TesseractOCRProvider

    return TesseractOCRProvider(**kwargs)


#: Registry engine: tên -> hàm khởi tạo.
PROVIDER_FACTORIES: dict[str, Callable[..., OCRProvider]] = {
    "paddle_vietocr": _load_paddle_vietocr,
    "tesseract": _load_tesseract,
}

#: Tham số nào thuộc về provider nào (factory bỏ qua kwargs không hợp lệ).
_COMMON_KWARGS = {"preprocess_config", "min_confidence"}
_PROVIDER_KWARGS: dict[str, set[str]] = {
    "paddle_vietocr": _COMMON_KWARGS
    | {"lang", "use_gpu", "vietocr_config", "vietocr_weights", "beamsearch", "enable_mkldnn"},
    "tesseract": _COMMON_KWARGS | {"lang", "psm", "oem"},
}


def available_engines() -> list[str]:
    """Danh sách engine thực sự chạy được trong môi trường hiện tại."""
    result: list[str] = []
    for name in ENGINE_PRIORITY:
        try:
            provider_cls_available = _is_engine_available(name)
        except Exception:  # pragma: no cover - phòng thủ
            provider_cls_available = False
        if provider_cls_available:
            result.append(name)
    return result


def _is_engine_available(name: str) -> bool:
    """Kiểm tra dependency của một engine mà không tải model."""
    if name == "paddle_vietocr":
        from .paddle_vietocr_provider import PaddleDetectorVietOCRProvider

        return PaddleDetectorVietOCRProvider.is_available()
    if name == "tesseract":
        from .tesseract_ocr import TesseractOCRProvider

        return TesseractOCRProvider.is_available()
    return False


def create_provider(
    engine: str = AUTO_ENGINE,
    *,
    strict: bool = False,
    **kwargs: Any,
) -> OCRProvider:
    """Tạo OCR provider theo tên, tự fallback khi thiếu dependency.

    :param engine: ``"paddle_vietocr"`` (mặc định của pipeline), ``"tesseract"``
        hoặc ``"auto"`` để thử lần lượt theo :data:`ENGINE_PRIORITY`.
    :param strict: nếu True thì không fallback - lỗi được raise ra ngoài.
    :raises ProviderUnavailableError: khi không provider nào dùng được.
    """
    requested = (engine or AUTO_ENGINE).strip().lower()
    if requested == AUTO_ENGINE:
        candidates = list(ENGINE_PRIORITY)
    elif requested in PROVIDER_FACTORIES:
        # Thử engine được yêu cầu trước, rồi tới các engine còn lại.
        candidates = [requested] + [e for e in ENGINE_PRIORITY if e != requested]
    else:
        raise ProviderUnavailableError(
            f"Engine không hợp lệ: {engine!r}. Hợp lệ: {', '.join(PROVIDER_FACTORIES)}"
        )

    errors: list[str] = []
    for index, name in enumerate(candidates):
        factory = PROVIDER_FACTORIES[name]
        allowed = _PROVIDER_KWARGS.get(name, set())
        provider_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        try:
            provider = factory(**provider_kwargs)
        except ProviderUnavailableError as exc:
            errors.append(f"{name}: {exc}")
            if strict and index == 0:
                raise
            logger.warning("Provider %s không khả dụng (%s)", name, exc)
            continue
        except Exception as exc:  # pragma: no cover - phòng thủ
            errors.append(f"{name}: {exc}")
            if strict and index == 0:
                raise ProviderUnavailableError(f"Khởi tạo provider {name} thất bại: {exc}") from exc
            logger.warning("Khởi tạo provider %s thất bại (%s)", name, exc)
            continue
        if index > 0:
            logger.warning("Đã fallback từ %r sang provider %r", requested, name)
        else:
            logger.info("Dùng OCR provider %r", name)
        return provider

    raise ProviderUnavailableError(
        "Không OCR provider nào khả dụng. Chi tiết: " + " | ".join(errors)
    )


__all__ = [
    "AUTO_ENGINE",
    "BoundingBox",
    "ENGINE_PRIORITY",
    "OCRError",
    "OCRProvider",
    "OCRResult",
    "OCRToken",
    "PROVIDER_FACTORIES",
    "PreprocessConfig",
    "ProviderUnavailableError",
    "available_engines",
    "create_provider",
    "group_tokens_into_lines",
    "sort_tokens_reading_order",
]
