"""Test tầng tương thích API của provider paddle_vietocr.

Các test này chạy được kể cả khi máy chưa cài paddleocr: chúng dùng object giả
mô phỏng đúng hành vi của paddleocr 2.x và 3.x, và gọi thẳng các hàm thuần logic
của provider thay vì khởi tạo model thật.
"""

from __future__ import annotations

import pytest

from ctkm_extractor.ocr.base import ProviderUnavailableError
from ctkm_extractor.ocr.paddle_vietocr_provider import (
    PaddleDetectorVietOCRProvider,
    looks_like_mkldnn_error,
    looks_like_pdx_reinit_error,
    reset_detector_cache,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    """Detector được cache theo process nên phải dọn giữa các test."""
    reset_detector_cache()
    yield
    reset_detector_cache()

#: Thông báo lỗi thật lấy từ log chạy trên Google Colab (paddlepaddle CPU + PIR).
ONEDNN_ERROR = (
    "(Unimplemented) ConvertPirAttribute2RuntimeAttribute not support "
    "[pir::ArrayAttribute<pir::DoubleAttribute>] "
    "(at /paddle/paddle/fluid/framework/new_executor/instruction/onednn/"
    "onednn_instruction.cc:116)"
)

SAMPLE_POLYGON = [[10.0, 20.0], [110.0, 20.0], [110.0, 50.0], [10.0, 50.0]]


def _bare_provider(**attrs: object) -> PaddleDetectorVietOCRProvider:
    """Provider không chạy ``__init__`` (không cần tải model thật)."""
    provider = object.__new__(PaddleDetectorVietOCRProvider)
    defaults = {
        "lang": "vi",
        "use_gpu": False,
        "enable_mkldnn": False,
        "_paddle_module": None,
        "_detector": None,
    }
    defaults.update(attrs)
    for key, value in defaults.items():
        setattr(provider, key, value)
    return provider


# ----------------------------------------------------------------------
# Detector giả
# ----------------------------------------------------------------------
class ModernDetector:
    """paddleocr 3.x: ``predict()`` chạy được, ``ocr()`` là alias không nhận det/rec."""

    def __init__(self, result: object = None) -> None:
        self.result = result if result is not None else [{"dt_polys": [SAMPLE_POLYGON]}]
        self.calls: list[str] = []

    def predict(self, image: object) -> object:
        self.calls.append("predict")
        return self.result

    def ocr(self, image: object, **kwargs: object) -> object:
        self.calls.append("ocr")
        if kwargs:
            raise TypeError(
                "PaddleOCR.predict() got an unexpected keyword argument "
                f"{next(iter(kwargs))!r}"
            )
        return self.result


class LegacyDetector:
    """paddleocr 2.x: chỉ có ``ocr(img, det=True, rec=False, cls=False)``."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def ocr(self, image: object, det: bool = True, rec: bool = True, cls: bool = True) -> object:
        self.calls.append(f"ocr(det={det},rec={rec},cls={cls})")
        return [[SAMPLE_POLYGON]]


class MkldnnBrokenDetector:
    """Detector dựng được nhưng chết ở backend oneDNN lúc inference."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image: object) -> object:
        self.calls += 1
        raise RuntimeError(ONEDNN_ERROR)

    def ocr(self, image: object, **kwargs: object) -> object:
        if kwargs:
            raise TypeError("PaddleOCR.predict() got an unexpected keyword argument 'det'")
        return self.predict(image)


# ----------------------------------------------------------------------
# Thứ tự gọi theo phiên bản API
# ----------------------------------------------------------------------
def test_modern_detector_khong_bi_goi_kem_det_rec():
    """paddleocr 3.x: phải gọi predict() trước, không truyền det/rec."""
    detector = ModernDetector()
    provider = _bare_provider(_detector=detector)

    polygons = provider._run_detection(object())

    assert detector.calls == ["predict"]
    assert polygons == [[(10.0, 20.0), (110.0, 20.0), (110.0, 50.0), (10.0, 50.0)]]


def test_legacy_detector_van_duoc_goi_kieu_2x():
    """paddleocr 2.x không có predict() nên phải dùng ocr(det=True, rec=False)."""
    detector = LegacyDetector()
    provider = _bare_provider(_detector=detector)

    polygons = provider._run_detection(object())

    assert detector.calls == ["ocr(det=True,rec=False,cls=False)"]
    assert len(polygons) == 1


def test_khong_goi_phuong_thuc_khong_ton_tai():
    """Detector chỉ có ocr() thì danh sách call không được chứa predict."""
    labels = [
        label
        for label, _ in PaddleDetectorVietOCRProvider._detection_calls(
            LegacyDetector(), object()
        )
    ]
    assert "predict" not in labels


# ----------------------------------------------------------------------
# Sự cố backend oneDNN
# ----------------------------------------------------------------------
def test_nhan_dien_loi_onednn():
    assert looks_like_mkldnn_error(ONEDNN_ERROR)
    assert not looks_like_mkldnn_error("Không tìm thấy ảnh: a.png")


def test_tu_tat_mkldnn_va_thu_lai():
    """Khi bật mkldnn mà backend chết, provider dựng lại detector với mkldnn tắt."""
    broken = MkldnnBrokenDetector()
    working = ModernDetector()
    rebuilt: list[bool] = []
    provider = _bare_provider(_detector=broken, enable_mkldnn=True)

    def fake_build(module, *, lang, use_gpu, enable_mkldnn):
        rebuilt.append(enable_mkldnn)
        return working

    provider._build_detector = fake_build  # type: ignore[method-assign]

    polygons = provider._run_detection(object())

    assert rebuilt == [False]
    assert provider.enable_mkldnn is False
    assert len(polygons) == 1


def test_khong_thu_lai_khi_mkldnn_da_tat():
    """Mkldnn đã tắt sẵn thì lỗi được báo ra luôn, kèm chi tiết để debug."""
    provider = _bare_provider(_detector=MkldnnBrokenDetector(), enable_mkldnn=False)

    with pytest.raises(ProviderUnavailableError) as excinfo:
        provider._run_detection(object())

    # Thông báo phải nêu rõ đã thử gì và lỗi gì, thay vì chỉ "không trả về kết quả".
    assert "onednn_instruction" in str(excinfo.value)
    assert "predict" in str(excinfo.value)


# ----------------------------------------------------------------------
# Khởi tạo detector
# ----------------------------------------------------------------------
class FakePaddleModule:
    """Module paddleocr giả với danh sách class được export tuỳ biến."""

    def __init__(self, *, text_detection=None, paddle_ocr=None) -> None:
        if text_detection is not None:
            self.TextDetection = text_detection
        if paddle_ocr is not None:
            self.PaddleOCR = paddle_ocr


def test_uu_tien_text_detection_cua_paddleocr_3x():
    seen: list[dict] = []

    class TextDetection:
        def __init__(self, **kwargs):
            seen.append(kwargs)

    module = FakePaddleModule(text_detection=TextDetection, paddle_ocr=object)
    detector = PaddleDetectorVietOCRProvider._build_detector(
        module, lang="vi", use_gpu=False, enable_mkldnn=False
    )

    assert isinstance(detector, TextDetection)
    # TextDetection raise "Unknown argument" nếu nhận lang, nên không được truyền.
    assert "lang" not in seen[0]
    assert seen[0] == {"device": "cpu", "enable_mkldnn": False}


def test_bo_qua_kwargs_khong_duoc_ho_tro_va_fallback():
    """Bản paddleocr không biết ``enable_mkldnn`` vẫn phải dựng được detector."""
    accepted: list[dict] = []

    class TextDetection:
        def __init__(self, **kwargs):
            if kwargs:
                raise ValueError(f"Unknown argument: {next(iter(kwargs))}")
            accepted.append(kwargs)

    module = FakePaddleModule(text_detection=TextDetection)
    detector = PaddleDetectorVietOCRProvider._build_detector(
        module, lang="vi", use_gpu=False, enable_mkldnn=False
    )

    assert isinstance(detector, TextDetection)
    assert accepted == [{}]


def test_paddleocr_2x_khong_nhan_det_rec_o_constructor():
    """Với bản 2.x, mọi bộ kwargs thử qua đều không được chứa det/rec."""
    module = FakePaddleModule(paddle_ocr=object)
    attempts = PaddleDetectorVietOCRProvider._detector_attempts(
        module, lang="vi", use_gpu=False, enable_mkldnn=False
    )

    assert attempts, "phải có ít nhất một cách khởi tạo"
    for _, _, kwargs in attempts:
        assert "det" not in kwargs
        assert "rec" not in kwargs


def test_bao_loi_ro_rang_khi_khong_dung_duoc_class_nao():
    class Broken:
        def __init__(self, **kwargs):
            raise RuntimeError("model không tải được")

    module = FakePaddleModule(text_detection=Broken, paddle_ocr=Broken)

    with pytest.raises(ProviderUnavailableError) as excinfo:
        PaddleDetectorVietOCRProvider._build_detector(
            module, lang="vi", use_gpu=False, enable_mkldnn=False
        )

    assert "model không tải được" in str(excinfo.value)


# ----------------------------------------------------------------------
# Chuẩn hoá output
# ----------------------------------------------------------------------
def test_chuan_hoa_output_dang_dict_3x():
    raw = [{"input_path": "a.png", "dt_polys": [SAMPLE_POLYGON], "dt_scores": [0.9]}]
    polygons = PaddleDetectorVietOCRProvider._normalize_detection_output(raw)
    assert polygons == [[(10.0, 20.0), (110.0, 20.0), (110.0, 50.0), (10.0, 50.0)]]


def test_chuan_hoa_output_dang_attribute():
    class Result:
        dt_polys = [SAMPLE_POLYGON]

    polygons = PaddleDetectorVietOCRProvider._normalize_detection_output([Result()])
    assert len(polygons) == 1


def test_chuan_hoa_output_dang_list_long_nhau_2x():
    raw = [[SAMPLE_POLYGON, SAMPLE_POLYGON]]
    polygons = PaddleDetectorVietOCRProvider._normalize_detection_output(raw)
    assert len(polygons) == 2


def test_chuan_hoa_output_rong_khong_crash():
    assert PaddleDetectorVietOCRProvider._normalize_detection_output([]) == []
    assert PaddleDetectorVietOCRProvider._normalize_detection_output(None) == []
    assert PaddleDetectorVietOCRProvider._normalize_detection_output([{"dt_polys": []}]) == []


# ----------------------------------------------------------------------
# PaddleX chỉ khởi tạo được một lần mỗi process
# ----------------------------------------------------------------------
#: Thông báo lỗi thật khi dựng provider thứ hai trong cùng một kernel notebook.
PDX_REINIT_ERROR = "PDX has already been initialized. Reinitialization is not supported."


def test_nhan_dien_loi_pdx_khoi_tao_lai():
    assert looks_like_pdx_reinit_error(PDX_REINIT_ERROR)
    assert not looks_like_pdx_reinit_error("model không tải được")


def test_detector_duoc_dung_lai_trong_cung_process():
    """Provider thứ hai phải dùng lại detector cũ thay vì dựng mới."""
    built: list[dict] = []

    class TextDetection:
        def __init__(self, **kwargs):
            if built:
                raise RuntimeError(PDX_REINIT_ERROR)
            built.append(kwargs)

    module = FakePaddleModule(text_detection=TextDetection)
    first = PaddleDetectorVietOCRProvider._build_detector(
        module, lang="vi", use_gpu=False, enable_mkldnn=False
    )
    second = PaddleDetectorVietOCRProvider._build_detector(
        module, lang="vi", use_gpu=False, enable_mkldnn=False
    )

    assert second is first
    assert len(built) == 1


def test_dung_lai_detector_ke_ca_khi_cau_hinh_khac():
    """Đổi tham số cũng không dựng lại được - PaddleX không cho khởi tạo lần hai."""
    calls: list[dict] = []

    class TextDetection:
        def __init__(self, **kwargs):
            if calls:
                raise RuntimeError(PDX_REINIT_ERROR)
            calls.append(kwargs)

    module = FakePaddleModule(text_detection=TextDetection)
    first = PaddleDetectorVietOCRProvider._build_detector(
        module, lang="vi", use_gpu=False, enable_mkldnn=False
    )
    second = PaddleDetectorVietOCRProvider._build_detector(
        module, lang="vi", use_gpu=False, enable_mkldnn=True
    )

    assert second is first
    assert len(calls) == 1


def test_khong_thu_tiep_sau_loi_khoi_tao_lai():
    """Lỗi PDX che mất nguyên nhân thật, nên phải dừng thử thay vì lặp lại nó."""
    attempts: list[dict] = []

    class TextDetection:
        def __init__(self, **kwargs):
            attempts.append(kwargs)
            raise RuntimeError(PDX_REINIT_ERROR)

    module = FakePaddleModule(text_detection=TextDetection, paddle_ocr=TextDetection)

    with pytest.raises(ProviderUnavailableError) as excinfo:
        PaddleDetectorVietOCRProvider._build_detector(
            module, lang="vi", use_gpu=False, enable_mkldnn=False
        )

    # Chỉ thử đúng một lần rồi dừng, thay vì lặp lại cùng một lỗi cho mọi cách.
    assert len(attempts) == 1
    assert "already been initialized" in str(excinfo.value)
