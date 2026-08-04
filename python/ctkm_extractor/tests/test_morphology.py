"""Test cho tầng dựng bảng bằng CV cổ điển (morphology).

Dùng ảnh bảng viền tổng hợp (vẽ bằng OpenCV) thay vì file ảnh thật, để test
độc lập, không cần mạng/file ngoài mà vẫn chạy qua đúng pipeline ảnh thật
(không phải mock). Việc tách cột từ token hàng header - thay vì gộp mọi
hàng - đã được kiểm chứng thêm bằng dữ liệu tọa độ thật trích từ file PDF
gốc trong quá trình review (xem lịch sử review), test ở đây tập trung vào
hành vi có thể tái lập trong CI.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cv2", reason="Cần OpenCV để test tầng morphology")
pytest.importorskip("numpy", reason="Cần numpy để test tầng morphology")

import numpy as np  # noqa: E402
import cv2  # noqa: E402

from ctkm_extractor.ocr.base import BoundingBox, OCRResult, OCRToken  # noqa: E402
from ctkm_extractor.table import STRATEGY_MORPHOLOGY, build_table  # noqa: E402
from ctkm_extractor.table.morphology import (  # noqa: E402
    MorphologyTableRecognizer,
    TableMorphologyUnavailableError,
    columns_in_band,
    detect_grid,
    detect_grid_segments,
    select_densest_band,
)


def _draw_bordered_table(
    row_boundaries: list[int], col_boundaries: list[int], size: tuple[int, int] = (400, 900)
) -> np.ndarray:
    """Vẽ 1 ảnh trắng có lưới bảng viền đen tại các toạ độ cho trước."""
    height, width = size
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    for y in row_boundaries:
        cv2.line(image, (0, y), (width, y), (0, 0, 0), thickness=2)
    for x in col_boundaries:
        cv2.line(image, (x, 0), (x, height), (0, 0, 0), thickness=2)
    return image


class TestDetectGrid:
    def test_raises_when_opencv_cannot_read_file(self) -> None:
        with pytest.raises(TableMorphologyUnavailableError):
            detect_grid("/khong/ton/tai/anh.jpg")

    def test_raises_on_unsupported_image_type(self) -> None:
        with pytest.raises(TableMorphologyUnavailableError):
            detect_grid(12345)  # type: ignore[arg-type]

    def test_detects_correct_row_and_column_count(self) -> None:
        # Bảng 2 hàng x 3 cột: 3 đường ngang, 4 đường dọc.
        rows = [40, 200, 360]
        cols = [60, 300, 550, 840]
        image = _draw_bordered_table(rows, cols)
        row_lines, col_lines = detect_grid(image)
        assert len(row_lines) == len(rows)
        assert len(col_lines) == len(cols)


class TestBuildTableImageSource:
    """``build_table`` phải dò đường kẻ trên ĐÚNG ảnh đã sinh ra token."""

    def test_uses_processed_image_instead_of_reloading_original(self, tmp_path) -> None:
        """Regression: tiền xử lý xoay/phóng to ảnh nên toạ độ token khác ảnh gốc.

        Trước khi sửa, ``build_table`` đọc lại ảnh gốc từ ``image_path``: với ảnh
        scan bị nghiêng thì không dò được đường kẻ ngang nào (hàng=0) nên
        morphology - mức dựng bảng tốt nhất - bị vô hiệu ngay trên loại ảnh cần
        nó nhất.
        """
        rows = [40, 140, 240]
        cols = [30, 400, 900]
        processed = _draw_bordered_table(rows, cols)

        # Ảnh "gốc" trên đĩa trắng trơn - không có đường kẻ nào để dò.
        original_path = tmp_path / "original.png"
        cv2.imwrite(str(original_path), np.full((400, 900, 3), 255, dtype=np.uint8))

        tokens = [
            OCRToken("Nội dung", BoundingBox(60, 70, 200, 100)),
            OCRToken("Chi tiết", BoundingBox(430, 70, 560, 100)),
            OCRToken("Chu kỳ", BoundingBox(60, 170, 200, 200)),
            OCRToken("tháng", BoundingBox(430, 170, 520, 200)),
        ]
        ocr_result = OCRResult(
            tokens=tokens,
            image_path=str(original_path),
            provider="fake",
            processed_image=processed,
        )

        result = build_table(ocr_result, use_pp_structure=False)

        assert result.strategy == STRATEGY_MORPHOLOGY
        assert result.table is not None
        assert result.table.n_cols == 2

    def test_falls_back_to_image_path_when_no_processed_image(self, tmp_path) -> None:
        """Không có ảnh tiền xử lý thì vẫn đọc ảnh gốc như trước."""
        image_path = tmp_path / "table.png"
        cv2.imwrite(str(image_path), _draw_bordered_table([40, 140, 240], [30, 400, 900]))

        tokens = [
            OCRToken("Nội dung", BoundingBox(60, 70, 200, 100)),
            OCRToken("Chi tiết", BoundingBox(430, 70, 560, 100)),
            OCRToken("Chu kỳ", BoundingBox(60, 170, 200, 200)),
            OCRToken("tháng", BoundingBox(430, 170, 520, 200)),
        ]
        ocr_result = OCRResult(tokens=tokens, image_path=str(image_path), provider="fake")

        result = build_table(ocr_result, use_pp_structure=False)

        assert result.strategy == STRATEGY_MORPHOLOGY


class TestMorphologyTableRecognizer:
    def test_is_available_true_when_opencv_installed(self) -> None:
        assert MorphologyTableRecognizer.is_available() is True

    def test_raises_when_too_few_lines(self) -> None:
        # Ảnh trắng trơn - không có đường kẻ nào.
        image = np.full((200, 400, 3), 255, dtype=np.uint8)
        recognizer = MorphologyTableRecognizer()
        with pytest.raises(TableMorphologyUnavailableError):
            recognizer.recognize(image, tokens=[])

    def test_recognizes_header_and_data_row_with_correct_columns(self) -> None:
        # Bảng 2 hàng (header + data) x 3 cột, giống cấu trúc bảng CTKM thật.
        rows = [40, 140, 240]
        cols = [30, 250, 500, 750]
        image = _draw_bordered_table(rows, cols)

        def token(text: str, x1: float, x2: float, y1: float, y2: float) -> OCRToken:
            return OCRToken(text=text, box=BoundingBox(x1, y1, x2, y2))

        tokens = [
            token("Mã gói", 60, 200, 70, 100),
            token("Phí đăng ký", 280, 460, 70, 100),
            token("Data GB", 530, 700, 70, 100),
            token("N180X", 60, 150, 170, 200),
            token("163636", 280, 380, 170, 200),
            token("60", 530, 570, 170, 200),
        ]

        recognizer = MorphologyTableRecognizer()
        table = recognizer.recognize(image, tokens)

        assert table.n_rows == 2
        assert table.n_cols == 3
        by_position = {(c.row, c.col): c.text for c in table.cells}
        assert by_position[(0, 0)] == "Mã gói"
        assert by_position[(0, 1)] == "Phí đăng ký"
        assert by_position[(0, 2)] == "Data GB"
        assert by_position[(1, 0)] == "N180X"
        assert by_position[(1, 1)] == "163636"
        assert by_position[(1, 2)] == "60"

    def test_raises_when_no_tokens_match_any_row(self) -> None:
        rows = [40, 140, 240]
        cols = [30, 250, 500, 750]
        image = _draw_bordered_table(rows, cols)
        # Token nằm ngoài mọi hàng đã dò (y quá lớn).
        far_token = OCRToken(text="lạc", box=BoundingBox(60, 900, 150, 930))
        recognizer = MorphologyTableRecognizer()
        with pytest.raises(TableMorphologyUnavailableError):
            recognizer.recognize(image, tokens=[far_token])

    def test_value_wider_than_header_stays_in_value_column(self) -> None:
        """Regression: token hàng dữ liệu tràn ra ngoài band cột của header.

        Band cột chỉ rộng bằng chữ trong header, nên giá trị dài hơn tiêu đề cột
        sẽ có phần không chồng lấn band nào. Trước khi sửa, những token đó rơi về
        cột 0 và bị dính vào ô nhãn ("Ưu đãi Data" nuốt "tối đa 8gb/1 ngày" ->
        dataGB đọc ra 8 thay vì 60).
        """
        rows = [40, 140, 240]
        cols = [30, 400, 900]
        image = _draw_bordered_table(rows, cols)

        def token(text: str, x1: float, x2: float, y1: float, y2: float) -> OCRToken:
            return OCRToken(text=text, box=BoundingBox(x1, y1, x2, y2))

        tokens = [
            token("Nội dung", 60, 200, 70, 100),
            # Header cột giá trị hẹp: chỉ từ x=430 tới x=560.
            token("Chi tiết", 430, 560, 70, 100),
            token("Ưu đãi Data", 60, 250, 170, 200),
            # Giá trị dài, phần đuôi nằm hoàn toàn bên phải band header.
            token("60GB/tháng,", 430, 600, 170, 200),
            token("tối đa 8gb/1 ngày", 620, 860, 170, 200),
        ]

        table = MorphologyTableRecognizer().recognize(image, tokens)

        by_position = {(c.row, c.col): c.text for c in table.cells}
        assert by_position[(1, 0)] == "Ưu đãi Data"
        assert by_position[(1, 1)] == "60GB/tháng, tối đa 8gb/1 ngày"

    def test_last_row_without_bottom_border_is_kept(self) -> None:
        """Hàng cuối nằm dưới đường kẻ cuối (ảnh thiếu viền dưới) vẫn được giữ."""
        rows = [40, 140, 240]
        cols = [30, 400, 900]
        image = _draw_bordered_table(rows, cols)

        def token(text: str, x1: float, x2: float, y1: float, y2: float) -> OCRToken:
            return OCRToken(text=text, box=BoundingBox(x1, y1, x2, y2))

        tokens = [
            token("Nội dung", 60, 200, 70, 100),
            token("Chi tiết", 430, 560, 70, 100),
            token("Chu kỳ", 60, 200, 170, 200),
            token("tháng", 430, 520, 170, 200),
            # Hàng cuối: nằm DƯỚI đường kẻ cuối cùng (y=240) nhưng còn trong
            # phạm vi một bước hàng, tức là hàng chưa được đóng viền.
            token("Gói cước áp dụng", 60, 300, 270, 300),
            token("Basic+, Family", 430, 640, 270, 300),
        ]

        table = MorphologyTableRecognizer().recognize(image, tokens)

        assert table.n_rows == 3
        by_position = {(c.row, c.col): c.text for c in table.cells}
        assert by_position[(2, 0)] == "Gói cước áp dụng"
        assert by_position[(2, 1)] == "Basic+, Family"

    def test_token_far_below_table_is_dropped(self) -> None:
        """Chữ ở chân trang (cách bảng quá xa) không bị kéo thành hàng của bảng."""
        rows = [40, 140, 240]
        cols = [30, 400, 900]
        image = _draw_bordered_table(rows, cols)

        def token(text: str, x1: float, x2: float, y1: float, y2: float) -> OCRToken:
            return OCRToken(text=text, box=BoundingBox(x1, y1, x2, y2))

        tokens = [
            token("Nội dung", 60, 200, 70, 100),
            token("Chi tiết", 430, 560, 70, 100),
            token("Chu kỳ", 60, 200, 170, 200),
            token("tháng", 430, 520, 170, 200),
            token("Trang 1/2 - biểu mẫu BM.12", 60, 400, 380, 399),
        ]

        table = MorphologyTableRecognizer().recognize(image, tokens)

        assert table.n_rows == 2
        assert all("Trang 1/2" not in cell.text for cell in table.cells)


def _draw_nested_table() -> np.ndarray:
    """Bảng NGOÀI 4 cột phủ cả trang, bên trong chứa bảng con 6 cột.

    Tái hiện cấu trúc biểu mẫu BM.12 thật: bảng CTKM nằm lồng trong một ô của
    bảng bao ngoài.
    """
    image = np.full((1200, 1600, 3), 255, dtype=np.uint8)
    # Bảng ngoài: 4 đường ngang, 4 đường dọc, phủ gần hết trang.
    for y in (100, 300, 800, 1100):
        cv2.line(image, (60, y), (1540, y), (0, 0, 0), 3)
    for x in (60, 200, 400, 1540):
        cv2.line(image, (x, 100), (x, 1100), (0, 0, 0), 3)
    # Bảng con nằm trong ô lớn (y 300..800): 3 đường ngang, 7 đường dọc.
    for y in (360, 520, 740):
        cv2.line(image, (430, y), (1500, y), (0, 0, 0), 3)
    for x in (430, 610, 790, 970, 1150, 1330, 1500):
        cv2.line(image, (x, 360), (x, 740), (0, 0, 0), 3)
    return image


class TestNestedTable:
    """Bảng lồng bảng: phải chọn đúng bảng con dày đường kẻ nhất."""

    def test_select_densest_band_tim_dung_bang_con(self) -> None:
        rows, segments = detect_grid_segments(_draw_nested_table())
        band = select_densest_band(rows, segments)

        assert band is not None
        top, bottom = band
        # Dải được chọn phải nằm gọn trong ô lớn của bảng ngoài (y 300..800).
        assert 300 <= top <= 380
        assert 700 <= bottom <= 800

    def test_select_densest_band_tra_none_khi_chi_co_mot_bang(self) -> None:
        rows, segments = detect_grid_segments(_draw_bordered_table([40, 140, 240], [30, 400, 900]))
        assert select_densest_band(rows, segments) is None

    def test_columns_in_band_loai_duong_ke_cua_bang_ngoai(self) -> None:
        rows, segments = detect_grid_segments(_draw_nested_table())
        band = select_densest_band(rows, segments)
        assert band is not None
        columns = columns_in_band(segments, *band)

        # 7 đường dọc của bảng con; 4 đường của bảng ngoài kéo dài vượt xa nên bị loại.
        assert len(columns) == 7
        assert min(columns) > 400

    def test_recognize_dung_bang_con_chu_khong_phai_bang_ngoai(self) -> None:
        image = _draw_nested_table()
        tokens = [
            # Text của bảng NGOÀI - phải bị loại.
            OCRToken("Chính sách được hưởng", BoundingBox(210, 400, 395, 440)),
            OCRToken("TT", BoundingBox(70, 150, 190, 190)),
            # Header + dữ liệu của bảng con (6 cột).
            OCRToken("Mã gói", BoundingBox(440, 400, 600, 440)),
            OCRToken("Phí ĐK", BoundingBox(620, 400, 780, 440)),
            OCRToken("Cước TB", BoundingBox(800, 400, 960, 440)),
            OCRToken("SMS", BoundingBox(980, 400, 1140, 440)),
            OCRToken("Data", BoundingBox(1160, 400, 1320, 440)),
            OCRToken("Chu kỳ", BoundingBox(1340, 400, 1490, 440)),
            OCRToken("N180X", BoundingBox(440, 600, 600, 640)),
            OCRToken("163636", BoundingBox(620, 600, 780, 640)),
            OCRToken("150534", BoundingBox(800, 600, 960, 640)),
            OCRToken("100", BoundingBox(980, 600, 1140, 640)),
            OCRToken("60GB", BoundingBox(1160, 600, 1320, 640)),
            OCRToken("tháng", BoundingBox(1340, 600, 1490, 640)),
        ]

        table = MorphologyTableRecognizer().recognize(image, tokens)

        assert table.n_rows == 2
        assert table.n_cols == 6  # cột của bảng CON, không phải 3 cột bảng ngoài
        by_position = {(c.row, c.col): c.text for c in table.cells}
        assert by_position[(0, 0)] == "Mã gói"
        assert by_position[(1, 0)] == "N180X"
        assert by_position[(1, 3)] == "100"
        # Text của bảng ngoài không được lọt vào bảng con.
        assert all("Chính sách" not in cell.text for cell in table.cells)
