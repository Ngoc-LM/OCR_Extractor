"""Provider mặc định: PP-OCRv5 text detector (DB) + VietOCR recognizer.

Tách detection và recognition thành 2 model riêng để tối ưu độ chính xác tiếng
Việt:

* **Detection** - ``paddleocr`` chạy ở chế độ chỉ detect (thuật toán DB,
  Differentiable Binarization của PP-OCRv5) để lấy polygon từng dòng/từ.
* **Recognition** - crop từng vùng đã detect (kèm perspective transform để nắn
  vùng nghiêng) rồi đưa qua ``vietocr`` (``vgg_transformer`` pretrained trên
  corpus tiếng Việt). Recognizer mặc định đa ngôn ngữ của PaddleOCR **không**
  được dùng.

Toàn bộ import nặng (``paddleocr``, ``vietocr``, ``torch``) nằm trong
``try/except ImportError`` tại thời điểm khởi tạo, nên module này luôn import
được; khi thiếu dependency thì raise :class:`ProviderUnavailableError` để factory
tự fallback sang Tesseract.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from .base import (
    BoundingBox,
    OCRProvider,
    OCRResult,
    OCRToken,
    ProviderUnavailableError,
)
from .preprocess import PreprocessConfig, is_available as cv_available, preprocess_image

logger = logging.getLogger(__name__)

#: Số pixel nới thêm quanh mỗi vùng crop, giúp recognizer không bị cụt dấu.
CROP_PADDING = 2

#: Dấu hiệu nhận biết lỗi đến từ backend oneDNN/mkldnn của paddlepaddle.
#:
#: Trên CPU, paddlepaddle mới (IR = PIR) có thể chết ngay lúc inference với
#: ``(Unimplemented) ConvertPirAttribute2RuntimeAttribute not support
#: [pir::ArrayAttribute<pir::DoubleAttribute>] (at .../onednn_instruction.cc)``.
#: Đây là lỗi backend chứ không phải lỗi API, và tắt oneDNN là workaround duy
#: nhất không cần đổi phiên bản paddlepaddle.
MKLDNN_ERROR_MARKERS = ("onednn", "mkldnn", "convertpirattribute")


#: Dấu hiệu PaddleX từ chối khởi tạo lần thứ hai trong cùng một process:
#: ``PDX has already been initialized. Reinitialization is not supported.``
PDX_REINIT_ERROR_MARKERS = ("already been initialized", "reinitialization is not supported")

#: Detector đã dựng được trong process này, dạng ``(key, detector)``.
#:
#: PaddleX chỉ cho phép khởi tạo **một lần cho mỗi process**, nên không thể dựng
#: detector thứ hai - kể cả khi chỉ muốn đổi tham số. Dùng lại detector cũ là
#: cách duy nhất để tạo được nhiều provider trong cùng một process (VD chạy A/B
#: hai cấu hình tiền xử lý trong một notebook).
_DETECTOR_CACHE: tuple[tuple[Any, ...], Any] | None = None


def looks_like_mkldnn_error(message: str) -> bool:
    """True nếu thông báo lỗi trông giống sự cố backend oneDNN của paddle."""
    lowered = message.lower()
    return any(marker in lowered for marker in MKLDNN_ERROR_MARKERS)


def looks_like_pdx_reinit_error(message: str) -> bool:
    """True nếu PaddleX từ chối vì đã được khởi tạo trước đó trong process này."""
    lowered = message.lower()
    return any(marker in lowered for marker in PDX_REINIT_ERROR_MARKERS)


def reset_detector_cache() -> None:
    """Xoá detector đang cache (dùng trong test)."""
    global _DETECTOR_CACHE
    _DETECTOR_CACHE = None


def _import_dependencies() -> tuple[Any, Any, Any, Any]:
    """Import các dependency nặng, gom lỗi vào ``ProviderUnavailableError``."""
    try:
        import numpy as np  # type: ignore
        import cv2  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
        raise ProviderUnavailableError(
            f"Thiếu numpy/opencv-python/Pillow cho provider paddle_vietocr: {exc}"
        ) from exc

    try:
        # Import cả module (không chỉ class PaddleOCR) vì paddleocr >= 3.0 còn
        # export class ``TextDetection`` - chạy riêng model DB, không tải kèm
        # recognizer đa ngôn ngữ mà pipeline này không dùng tới.
        import paddleocr  # type: ignore
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
        raise ProviderUnavailableError(
            f"Thiếu paddleocr/paddlepaddle cho text detection: {exc}"
        ) from exc

    return np, cv2, Image, paddleocr


def _import_vietocr() -> tuple[Any, Any]:
    """Import VietOCR (kéo theo ``torch``)."""
    try:
        from vietocr.tool.config import Cfg  # type: ignore
        from vietocr.tool.predictor import Predictor  # type: ignore
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
        raise ProviderUnavailableError(
            f"Thiếu vietocr/torch cho recognition tiếng Việt: {exc}"
        ) from exc
    return Cfg, Predictor


class PaddleDetectorVietOCRProvider(OCRProvider):
    """Detect bằng PP-OCRv5 (DB), recognize bằng VietOCR ``vgg_transformer``."""

    name = "paddle_vietocr"

    def __init__(
        self,
        *,
        lang: str = "vi",
        use_gpu: bool = False,
        vietocr_config: str = "vgg_transformer",
        vietocr_weights: str | None = None,
        beamsearch: bool = False,
        preprocess_config: PreprocessConfig | None = None,
        min_confidence: float = 0.0,
        enable_mkldnn: bool = False,
    ) -> None:
        """Khởi tạo detector + recognizer.

        :param enable_mkldnn: bật backend oneDNN của paddlepaddle trên CPU. Mặc
            định **tắt** vì tổ hợp paddlepaddle PIR + oneDNN hay chết lúc
            inference (xem :data:`MKLDNN_ERROR_MARKERS`); bật lên thì chạy nhanh
            hơn nhưng nếu lỗi, provider tự tắt và thử lại một lần.

        Raise :class:`ProviderUnavailableError` nếu thiếu dependency hoặc không tải
        được model, để tầng factory fallback sang provider nhẹ hơn.
        """
        self._np, self._cv2, self._pil_image, paddle_module = _import_dependencies()
        cfg_loader, predictor_cls = _import_vietocr()

        self.lang = lang
        self.use_gpu = use_gpu
        self.min_confidence = min_confidence
        self.preprocess_config = preprocess_config or PreprocessConfig()
        self.enable_mkldnn = enable_mkldnn

        self._paddle_module = paddle_module
        self._detector = self._build_detector(
            paddle_module, lang=lang, use_gpu=use_gpu, enable_mkldnn=enable_mkldnn
        )
        self._recognizer = self._build_recognizer(
            cfg_loader,
            predictor_cls,
            config_name=vietocr_config,
            weights=vietocr_weights,
            use_gpu=use_gpu,
            beamsearch=beamsearch,
        )

    # ------------------------------------------------------------------
    # Khởi tạo model
    # ------------------------------------------------------------------
    @staticmethod
    def _detector_attempts(
        paddle_module: Any, *, lang: str, use_gpu: bool, enable_mkldnn: bool
    ) -> list[tuple[str, Any, dict[str, Any]]]:
        """Danh sách cách khởi tạo detector, ưu tiên API mới nhất.

        Trả về các bộ ``(nhãn, class, kwargs)``. Tách riêng khỏi
        :meth:`_build_detector` để test được thứ tự mà không cần cài paddleocr.
        """
        device = "gpu" if use_gpu else "cpu"
        attempts: list[tuple[str, Any, dict[str, Any]]] = []

        # paddleocr >= 3.0: module TextDetection chỉ nạp đúng model DB. Không
        # nhận ``lang`` (parse_common_args raise "Unknown argument") và cũng
        # không tải model recognition 77 MB mà pipeline này không dùng.
        text_detection_cls = getattr(paddle_module, "TextDetection", None)
        if text_detection_cls is not None:
            attempts.append(
                ("TextDetection", text_detection_cls,
                 {"device": device, "enable_mkldnn": enable_mkldnn})
            )
            attempts.append(("TextDetection", text_detection_cls, {}))

        paddle_ocr_cls = getattr(paddle_module, "PaddleOCR", None)
        if paddle_ocr_cls is not None:
            # paddleocr 3.x: pipeline đầy đủ, tắt hết bước phụ cho nhẹ.
            attempts.append(
                ("PaddleOCR", paddle_ocr_cls, {
                    "lang": lang,
                    "device": device,
                    "enable_mkldnn": enable_mkldnn,
                    "use_textline_orientation": False,
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                })
            )
            attempts.append(
                ("PaddleOCR", paddle_ocr_cls, {
                    "lang": lang,
                    "use_textline_orientation": False,
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                })
            )
            # paddleocr 2.x: det/rec là tham số của ``ocr()`` chứ không phải của
            # constructor, nên ở đây chỉ truyền tham số constructor hợp lệ.
            legacy: dict[str, Any] = {
                "lang": lang, "use_angle_cls": False, "show_log": False
            }
            if use_gpu:
                legacy["use_gpu"] = True
            attempts.append(("PaddleOCR", paddle_ocr_cls, legacy))
            attempts.append(("PaddleOCR", paddle_ocr_cls, {"lang": lang}))
            attempts.append(("PaddleOCR", paddle_ocr_cls, {}))
        return attempts

    @classmethod
    def _build_detector(
        cls,
        paddle_module: Any,
        *,
        lang: str,
        use_gpu: bool,
        enable_mkldnn: bool,
    ) -> Any:
        """Tạo text detector, tương thích cả paddleocr 2.x lẫn 3.x.

        Detector được cache theo process: PaddleX không cho khởi tạo lần thứ hai
        nên provider thứ hai trong cùng process phải dùng lại detector đã có.
        """
        global _DETECTOR_CACHE

        key = (lang, use_gpu, enable_mkldnn)
        if _DETECTOR_CACHE is not None:
            cached_key, cached_detector = _DETECTOR_CACHE
            if cached_key != key:
                logger.warning(
                    "PaddleX chỉ khởi tạo được một lần mỗi process: dùng lại detector "
                    "đã dựng với %s thay vì %s",
                    cached_key,
                    key,
                )
            return cached_detector

        attempts = cls._detector_attempts(
            paddle_module, lang=lang, use_gpu=use_gpu, enable_mkldnn=enable_mkldnn
        )
        if not attempts:
            raise ProviderUnavailableError(
                "paddleocr không export class TextDetection lẫn PaddleOCR"
            )
        errors: list[str] = []
        for label, factory, kwargs in attempts:
            try:
                detector = factory(**kwargs)
            except Exception as exc:
                errors.append(f"{label}({_format_kwargs(kwargs)}): {exc}")
                logger.debug("Khởi tạo %s với %s thất bại: %s", label, kwargs, exc)
                if looks_like_pdx_reinit_error(str(exc)):
                    # Lần khởi tạo đầu đã chiếm chỗ của PaddleX rồi; mọi cách còn
                    # lại cũng sẽ báo đúng lỗi này và che mất nguyên nhân thật.
                    logger.debug("PaddleX đã được khởi tạo trước đó, dừng thử tiếp")
                    break
                continue
            logger.debug("Text detector: %s(%s)", label, _format_kwargs(kwargs))
            _DETECTOR_CACHE = (key, detector)
            return detector
        raise ProviderUnavailableError(
            "Không khởi tạo được text detector PaddleOCR. Đã thử: " + " | ".join(errors)
        )

    @staticmethod
    def _build_recognizer(
        cfg_loader: Any,
        predictor_cls: Any,
        *,
        config_name: str,
        weights: str | None,
        use_gpu: bool,
        beamsearch: bool,
    ) -> Any:
        """Tạo ``vietocr.Predictor`` với config pretrained tiếng Việt."""
        try:
            config = cfg_loader.load_config_from_name(config_name)
            config["device"] = "cuda:0" if use_gpu else "cpu"
            config["predictor"]["beamsearch"] = beamsearch
            if weights:
                config["weights"] = weights
            config["cnn"]["pretrained"] = False
            return predictor_cls(config)
        except Exception as exc:  # pragma: no cover - phụ thuộc mạng/model
            raise ProviderUnavailableError(
                f"Không khởi tạo được VietOCR ({config_name}): {exc}"
            ) from exc

    @classmethod
    def is_available(cls) -> bool:
        """Kiểm tra nhanh dependency mà không tải model."""
        try:
            _import_dependencies()
            _import_vietocr()
        except ProviderUnavailableError:
            return False
        return True

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    @staticmethod
    def _detection_calls(detector: Any, image: Any) -> list[tuple[str, Any]]:
        """Các cách gọi detector, xếp theo API mà object thực sự hỗ trợ.

        paddleocr 3.x dùng ``predict()``; ``ocr()`` chỉ còn là alias deprecated
        và **không** nhận ``det``/``rec`` nữa (gọi kèm sẽ báo
        ``predict() got an unexpected keyword argument 'det'``). paddleocr 2.x
        thì ngược lại: chỉ có ``ocr(img, det=True, rec=False, cls=False)``.
        Danh sách vẫn giữ đủ mọi biến thể để không phụ thuộc vào việc đoán đúng
        phiên bản, chỉ khác thứ tự thử.
        """
        modern = [
            ("predict", lambda: detector.predict(image)),
            ("ocr", lambda: detector.ocr(image)),
        ]
        legacy = [
            ("ocr(det,rec,cls)", lambda: detector.ocr(image, det=True, rec=False, cls=False)),
            ("ocr(det,rec)", lambda: detector.ocr(image, det=True, rec=False)),
            ("ocr", lambda: detector.ocr(image)),
        ]
        calls = modern + legacy if hasattr(detector, "predict") else legacy + modern
        return [(label, call) for label, call in calls if hasattr(detector, label.split("(")[0])]

    def _detect_once(self, image: Any) -> tuple[Any, list[str]]:
        """Gọi detector một lượt, trả ``(raw_output, danh_sách_lỗi)``."""
        errors: list[str] = []
        for label, call in self._detection_calls(self._detector, image):
            try:
                raw = call()
            except Exception as exc:
                errors.append(f"{label}: {exc}")
                logger.debug("Gọi detector qua %s thất bại: %s", label, exc)
                continue
            if raw is not None:
                return raw, errors
            errors.append(f"{label}: trả về None")
        return None, errors

    def _run_detection(self, image: Any) -> list[list[tuple[float, float]]]:
        """Trả về danh sách polygon 4 điểm cho từng vùng text detect được."""
        raw, errors = self._detect_once(image)
        if raw is None and self.enable_mkldnn and any(
            looks_like_mkldnn_error(message) for message in errors
        ):
            # Backend oneDNN chết giữa chừng: dựng lại detector với run_mode
            # "paddle" thuần rồi thử đúng một lần nữa.
            logger.warning(
                "Text detector lỗi ở backend oneDNN, thử lại với enable_mkldnn=False"
            )
            self.enable_mkldnn = False
            self._detector = self._build_detector(
                self._paddle_module,
                lang=self.lang,
                use_gpu=self.use_gpu,
                enable_mkldnn=False,
            )
            raw, retry_errors = self._detect_once(image)
            errors.extend(retry_errors)
        if raw is None:
            raise ProviderUnavailableError(
                "Text detector không trả về kết quả. Đã thử: " + " | ".join(errors)
            )
        return self._normalize_detection_output(raw)

    @staticmethod
    def _normalize_detection_output(raw: Any) -> list[list[tuple[float, float]]]:
        """Chuẩn hoá output detect về ``list[polygon]`` cho mọi phiên bản paddleocr."""
        polygons: list[list[tuple[float, float]]] = []
        # Khoá chứa polygon, xếp theo thứ tự ưu tiên. ``dt_polys`` là khoá của
        # TextDetResult/OCRResult (paddleocr 3.x - đều là subclass của dict).
        poly_keys = ("dt_polys", "boxes", "rec_polys", "dt_boxes")

        def add_polygon(candidate: Any) -> None:
            try:
                points = [(float(p[0]), float(p[1])) for p in candidate]
            except (TypeError, ValueError, IndexError):
                return
            if len(points) >= 3:
                polygons.append(points)

        def walk(node: Any) -> None:
            if node is None:
                return
            # dict: PaddleOCR 3.x trả {'dt_polys': [...]} hoặc object có .json
            if isinstance(node, dict):
                for key in poly_keys:
                    if key in node:
                        for item in node[key]:
                            add_polygon(item)
                        return
                for value in node.values():
                    walk(value)
                return
            # Phòng trường hợp result object không phải dict subclass mà chỉ phơi
            # polygon qua attribute.
            for key in poly_keys:
                value = getattr(node, key, None)
                if value is not None and not callable(value):
                    for item in value:
                        add_polygon(item)
                    return
            if hasattr(node, "json"):  # pragma: no cover - phụ thuộc phiên bản
                try:
                    walk(node.json)
                    return
                except Exception:
                    pass
            if isinstance(node, (list, tuple)):
                if node and _looks_like_polygon(node):
                    add_polygon(node)
                    return
                for item in node:
                    walk(item)
                return
            if hasattr(node, "tolist"):
                walk(node.tolist())

        walk(raw)
        return polygons

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------
    def _crop(self, image: Any, polygon: Sequence[tuple[float, float]]) -> Any | None:
        """Cắt vùng text và nắn phối cảnh về hình chữ nhật."""
        np = self._np
        cv2 = self._cv2
        try:
            points = np.array(polygon, dtype="float32")
            if points.shape[0] != 4:
                box = BoundingBox.from_polygon(polygon)
                points = np.array(
                    [
                        [box.x1, box.y1],
                        [box.x2, box.y1],
                        [box.x2, box.y2],
                        [box.x1, box.y2],
                    ],
                    dtype="float32",
                )
            points = _order_points(np, points)
            width = int(
                max(
                    np.linalg.norm(points[0] - points[1]),
                    np.linalg.norm(points[3] - points[2]),
                )
            )
            height = int(
                max(
                    np.linalg.norm(points[0] - points[3]),
                    np.linalg.norm(points[1] - points[2]),
                )
            )
            width += 2 * CROP_PADDING
            height += 2 * CROP_PADDING
            if width < 4 or height < 4:
                return None
            destination = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype="float32",
            )
            matrix = cv2.getPerspectiveTransform(points, destination)
            warped = cv2.warpPerspective(
                image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
            )
            # Ảnh dọc (chữ xoay 90 độ) thì xoay lại cho recognizer.
            if height > width * 1.5:
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            return warped
        except Exception as exc:  # pragma: no cover - phòng thủ
            logger.warning("Crop vùng text thất bại: %s", exc)
            return None

    def _recognize(self, crop: Any) -> tuple[str, float]:
        """Nhận dạng 1 vùng crop bằng VietOCR, trả ``(text, confidence)``."""
        cv2 = self._cv2
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_image = self._pil_image.fromarray(rgb)
        except Exception as exc:  # pragma: no cover - phòng thủ
            logger.warning("Chuyển crop sang PIL thất bại: %s", exc)
            return "", 0.0
        try:
            output = self._recognizer.predict(pil_image, return_prob=True)
        except TypeError:  # pragma: no cover - phiên bản vietocr cũ
            try:
                output = self._recognizer.predict(pil_image)
            except Exception as exc:
                logger.warning("VietOCR nhận dạng thất bại: %s", exc)
                return "", 0.0
        except Exception as exc:  # pragma: no cover - phòng thủ
            logger.warning("VietOCR nhận dạng thất bại: %s", exc)
            return "", 0.0

        if isinstance(output, tuple) and len(output) == 2:
            text, probability = output
            try:
                confidence = float(probability)
            except (TypeError, ValueError):
                confidence = 1.0
            return str(text), confidence
        return str(output), 1.0

    # ------------------------------------------------------------------
    # API công khai
    # ------------------------------------------------------------------
    def extract(self, image_path: str) -> OCRResult:
        """Chạy detect + recognize và trả về danh sách token có toạ độ."""
        warnings: list[str] = []
        if cv_available():
            processed, ok = preprocess_image(image_path, self.preprocess_config)
            if not ok:
                warnings.append("Không tiền xử lý được ảnh, dùng ảnh gốc")
            image = processed if not isinstance(processed, str) else None
        else:  # pragma: no cover - phụ thuộc môi trường
            image = None
        if image is None:  # pragma: no cover - phụ thuộc môi trường
            image = self._cv2.imread(image_path)
        if image is None:
            raise ProviderUnavailableError(f"Không đọc được ảnh: {image_path}")

        height, width = image.shape[:2]
        polygons = self._run_detection(image)
        if not polygons:
            warnings.append("Detector không tìm thấy vùng text nào")
            logger.warning("PP-OCR detector không tìm thấy vùng text trong %s", image_path)

        tokens: list[OCRToken] = []
        for polygon in polygons:
            crop = self._crop(image, polygon)
            if crop is None:
                continue
            text, confidence = self._recognize(crop)
            if not text.strip():
                continue
            if confidence < self.min_confidence:
                logger.debug("Bỏ token confidence thấp %.3f: %r", confidence, text)
                continue
            tokens.append(
                OCRToken(
                    text=text,
                    box=BoundingBox.from_polygon(polygon),
                    confidence=confidence,
                    polygon=tuple((float(p[0]), float(p[1])) for p in polygon),
                )
            )

        return OCRResult(
            tokens=tokens,
            image_path=image_path,
            image_size=(int(width), int(height)),
            provider=self.name,
            warnings=warnings,
            processed_image=image,
        )


def _format_kwargs(kwargs: dict[str, Any]) -> str:
    """Rút gọn kwargs cho thông báo lỗi/log."""
    return ", ".join(f"{key}={value!r}" for key, value in kwargs.items())


def _looks_like_polygon(node: Sequence[Any]) -> bool:
    """Heuristic: node là list các điểm ``[x, y]``."""
    if len(node) < 3:
        return False
    for item in node:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            if hasattr(item, "tolist"):
                converted = item.tolist()
                if isinstance(converted, (list, tuple)) and len(converted) == 2:
                    continue
            return False
        try:
            float(item[0])
            float(item[1])
        except (TypeError, ValueError):
            return False
    return True


def _order_points(np: Any, points: Any) -> Any:
    """Sắp 4 điểm theo thứ tự trên-trái, trên-phải, dưới-phải, dưới-trái."""
    ordered = np.zeros((4, 2), dtype="float32")
    total = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    ordered[0] = points[np.argmin(total)]
    ordered[2] = points[np.argmax(total)]
    ordered[1] = points[np.argmin(diff)]
    ordered[3] = points[np.argmax(diff)]
    return ordered
