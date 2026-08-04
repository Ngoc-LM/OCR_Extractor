"""Test orchestrator end-to-end với OCR được mock bằng raw-text blob.

Không cần ảnh thật, không cần paddleocr/vietocr: một provider giả lập trả về
``OCRResult`` dựng từ text fixture, đúng như OCR thật sẽ trả về.

Giá trị trong fixture lấy theo ảnh mẫu của đề bài - chúng chỉ tồn tại ở tầng test,
logic trích xuất không hard-code bất kỳ giá trị nào trong số đó.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ctkm_extractor.cli import main
from ctkm_extractor.extraction.extractor import (
    ExtractionResult,
    FieldResult,
    cut_at_next_label,
    merge_page_results,
    SOURCE_MISSING,
    CTKMExtractor,
    SchemaError,
    load_schema,
)
from ctkm_extractor.ocr import PreprocessConfig
from ctkm_extractor.ocr.base import BoundingBox, OCRProvider, OCRResult, OCRToken
from ctkm_extractor.table import (
    STRATEGY_CLUSTER,
    STRATEGY_PP_STRUCTURE,
    STRATEGY_RAW_TEXT,
    PPStructureTableRecognizer,
    Table,
    TableStructureUnavailableError,
    assign_tokens_to_cells,
    build_table,
)
from ctkm_extractor.table.pp_structure import parse_table_html
from ctkm_extractor.table.reconstruct import table_from_rows

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
SAMPLE_BLOB = """
CHƯƠNG TRÌNH KHUYẾN MẠI
Nội dung Chi tiết
Tên mã giá OCS CTKMN180X
Cước đăng ký (chưa VAT) 163,636.3636
Cước thuê bao tháng 0
Ưu đãi thoại nội mạng MP 20p đầu tiên/cuộc gọi
Ưu đãi thoại ngoại mạng 150 phút
Ưu đãi SMS 100 tin nhắn
Ưu đãi Data 60GB/tháng, tối đa 8gb/1 ngày
Youtube: 25gb/tháng
Spotify: 25gb/tháng
Chu kỳ gia hạn: tháng
Gói cước được phép ĐK: Basic+, Family, Corporate++
"""

# Cùng nội dung nhưng đổi tên nhãn và thứ tự dòng - chứng minh schema-driven,
# không phụ thuộc mẫu bảng cụ thể.
ALTERNATIVE_BLOB = """
Chu kỳ: tháng
Mã gói cước: GOI_XYZ99
SMS: 50
Phí đăng ký: 99.000
Cước thuê bao: 55.000
Thoại ngoại mạng: 30 phút
Data tốc độ cao: 5GB
"""

# Ảnh mờ, OCR mất chữ và sai ký tự: chương trình phải trả null chứ không crash.
NOISY_BLOB = """
CHUONG TRINH KHUYEN MAI
Ten ma gla 0CS
??? ---
Cuoc dang ky
"""

EXPECTED_FIELDS = [
    "packageCode",
    "registerFee",
    "monthlyFee",
    "onnetMinutes",
    "offnetMinutes",
    "sms",
    "dataGB",
    "youtubeGB",
    "spotifyGB",
    "cycle",
    "allowedPackages",
]


class FakeOCRProvider(OCRProvider):
    """Provider giả lập: trả token dựng từ một blob text cố định."""

    name = "fake"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[str] = []

    @classmethod
    def is_available(cls) -> bool:
        return True

    def extract(self, image_path: str) -> OCRResult:
        self.calls.append(image_path)
        result = OCRResult.from_text(self.text, provider=self.name)
        result.image_path = image_path
        return result


class BrokenOCRProvider(OCRProvider):
    """Provider luôn lỗi - dùng để kiểm tra pipeline không crash."""

    name = "broken"

    def extract(self, image_path: str) -> OCRResult:
        raise RuntimeError("model weights hỏng")


@pytest.fixture()
def extractor() -> CTKMExtractor:
    """Extractor dùng schema mặc định, không gọi PP-Structure (không có ảnh thật)."""
    return CTKMExtractor(provider=FakeOCRProvider(SAMPLE_BLOB), use_pp_structure=False)


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
class TestSchema:
    def test_default_schema_declares_all_required_fields(self) -> None:
        schema = load_schema()
        assert schema.field_names == EXPECTED_FIELDS

    def test_every_field_has_parser_and_aliases(self) -> None:
        for spec in load_schema().fields:
            assert spec.parser, f"{spec.name} thiếu parser"
            assert spec.aliases, f"{spec.name} thiếu aliases"

    def test_missing_schema_file_raises_schema_error(self, tmp_path: Path) -> None:
        with pytest.raises(SchemaError):
            load_schema(tmp_path / "khong_ton_tai.yaml")

    def test_invalid_schema_raises_schema_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("version: 1\nfields: []\n", encoding="utf-8")
        with pytest.raises(SchemaError):
            load_schema(bad)


# --------------------------------------------------------------------------
# End-to-end với OCR mock
# --------------------------------------------------------------------------
class TestEndToEnd:
    def test_output_has_exactly_schema_fields_in_order(self, extractor: CTKMExtractor) -> None:
        payload = extractor.extract_from_image("sample.jpg").to_dict()
        assert list(payload.keys()) == EXPECTED_FIELDS

    def test_extracts_expected_values(self, extractor: CTKMExtractor) -> None:
        payload = extractor.extract_from_image("sample.jpg").to_dict()
        assert payload["packageCode"] == "CTKMN180X"
        assert payload["registerFee"] == pytest.approx(163636.3636)
        assert payload["monthlyFee"] == 0
        # onnetMinutes là mô tả chính sách ("MP 20p đầu tiên..."), không phải số
        # phút thuần - xem ghi chú trong schema.yaml.
        assert payload["onnetMinutes"] == "MP 20p đầu tiên/cuộc gọi"
        assert payload["offnetMinutes"] == 150
        assert payload["sms"] == 100
        assert payload["dataGB"] == 60
        assert payload["youtubeGB"] == 25
        assert payload["spotifyGB"] == 25
        assert payload["cycle"] == "tháng"
        assert payload["allowedPackages"] == ["Basic+", "Family", "Corporate++"]

    def test_value_types_match_schema(self, extractor: CTKMExtractor) -> None:
        payload = extractor.extract_from_image("sample.jpg").to_dict()
        assert isinstance(payload["packageCode"], str)
        assert isinstance(payload["registerFee"], (int, float))
        assert isinstance(payload["sms"], int)
        assert isinstance(payload["allowedPackages"], list)
        assert all(isinstance(item, str) for item in payload["allowedPackages"])

    def test_result_is_json_serialisable(self, extractor: CTKMExtractor) -> None:
        result = extractor.extract_from_image("sample.jpg")
        decoded = json.loads(result.to_json())
        assert decoded["packageCode"] == "CTKMN180X"
        assert "tháng" in result.to_json()  # không escape tiếng Việt

    def test_provider_receives_image_path(self) -> None:
        provider = FakeOCRProvider(SAMPLE_BLOB)
        CTKMExtractor(provider=provider, use_pp_structure=False).extract_from_image("x.png")
        assert provider.calls == ["x.png"]

    def test_debug_report_contains_raw_text_and_table(self, extractor: CTKMExtractor) -> None:
        report = extractor.extract_from_image("sample.jpg").debug_report()
        assert "OCR raw text" in report
        assert "Bảng đã dựng" in report
        assert "CTKMN180X" in report

    def test_alternative_template_uses_other_aliases(self) -> None:
        provider = FakeOCRProvider(ALTERNATIVE_BLOB)
        payload = (
            CTKMExtractor(provider=provider, use_pp_structure=False)
            .extract_from_image("other.jpg")
            .to_dict()
        )
        assert payload["packageCode"] == "GOI_XYZ99"
        assert payload["registerFee"] == 99000
        assert payload["monthlyFee"] == 55000
        assert payload["offnetMinutes"] == 30
        assert payload["sms"] == 50
        assert payload["dataGB"] == 5
        assert payload["cycle"] == "tháng"
        # Không có trong mẫu này -> null chứ không phải lỗi.
        assert payload["youtubeGB"] is None
        assert payload["allowedPackages"] is None

    def test_real_bm12_document_headers(self) -> None:
        """Regression test dùng ĐÚNG NGUYÊN VĂN header/giá trị của ảnh mẫu BM.12.

        Không dùng nhãn đã "làm đẹp" khớp sẵn alias như ``SAMPLE_BLOB`` - đây
        chính là cách phát hiện ra 4 bug thực tế (monthlyFee/allowedPackages
        null, onnetMinutes lấy nhầm cột offnetMinutes, dataGB đọc nhầm
        "(TK533)" trong header thay vì giá trị thật bên dưới).
        """
        header = [
            "Tên mã giá OCS",
            "Phí đăng ký (VNĐ/tháng)",
            "Cước TB (VND tháng)",
            "TK thoại",
            "Thoại ngoại mạng",
            "SMS Trong nước",
            "Lưu lượng Data đa hướng (TK533)",
            "Ưu đãi data đơn",
        ]
        data = [
            "CTKMN180X",
            "163,636.3636",
            "150,534.213",
            "MP 20p đầu tiên",
            "150",
            "100",
            "60GB/tháng, tối đa 8gb/1 ngày",
            "Youtube: 25gb/tháng\nSpotify: 25GB/ tháng",
        ]
        table = table_from_rows([header, data])
        raw_text = (
            "Chu kỳ gia hạn: tháng.\n"
            "Gói cước chính được đăng kí: Basic+, Family, Corporate++"
        )
        payload = CTKMExtractor().extract_from_table(table, raw_text=raw_text).to_dict()

        assert payload["packageCode"] == "CTKMN180X"
        assert payload["registerFee"] == pytest.approx(163636.3636)
        assert payload["monthlyFee"] == pytest.approx(150534.213)
        assert payload["onnetMinutes"] == "MP 20p đầu tiên"
        assert payload["offnetMinutes"] == 150
        assert payload["sms"] == 100
        assert payload["dataGB"] == 60
        assert payload["youtubeGB"] == 25
        assert payload["spotifyGB"] == 25
        assert payload["cycle"] == "tháng"
        assert payload["allowedPackages"] == ["Basic+", "Family", "Corporate++"]


# --------------------------------------------------------------------------
# Xử lý dữ liệu không chuẩn
# --------------------------------------------------------------------------
class TestRobustness:
    def test_noisy_input_returns_nulls_without_crashing(self) -> None:
        provider = FakeOCRProvider(NOISY_BLOB)
        result = CTKMExtractor(provider=provider, use_pp_structure=False).extract_from_image(
            "noisy.jpg"
        )
        payload = result.to_dict()
        assert list(payload.keys()) == EXPECTED_FIELDS
        assert payload["sms"] is None
        assert payload["allowedPackages"] is None
        assert result.warnings, "phải ghi cảnh báo cho các field thiếu"

    def test_empty_ocr_output_returns_all_nulls(self) -> None:
        provider = FakeOCRProvider("")
        result = CTKMExtractor(provider=provider, use_pp_structure=False).extract_from_image(
            "blank.jpg"
        )
        assert result.to_dict() == {name: None for name in EXPECTED_FIELDS}
        assert result.table_strategy == "raw_text"

    def test_ocr_failure_is_caught(self) -> None:
        result = CTKMExtractor(
            provider=BrokenOCRProvider(), use_pp_structure=False
        ).extract_from_image("broken.jpg")
        assert result.to_dict() == {name: None for name in EXPECTED_FIELDS}
        assert any("OCR thất bại" in warning for warning in result.warnings)

    def test_missing_field_is_marked_missing(self) -> None:
        result = CTKMExtractor(
            provider=FakeOCRProvider("Chu kỳ: tháng"), use_pp_structure=False
        ).extract_from_image("partial.jpg")
        assert result.fields["cycle"].value == "tháng"
        assert result.fields["sms"].value is None
        assert result.fields["sms"].source == SOURCE_MISSING

    def test_ocr_character_noise_still_parses_numbers(self) -> None:
        # "l" thay cho "1" và "O" thay cho "0" ngay trong cụm số.
        blob = "Cước đăng ký (chưa VAT) l63,636.3636\nƯu đãi SMS 1OO"
        payload = (
            CTKMExtractor(provider=FakeOCRProvider(blob), use_pp_structure=False)
            .extract_from_image("ocr_noise.jpg")
            .to_dict()
        )
        assert payload["registerFee"] == pytest.approx(163636.3636)
        assert payload["sms"] == 100


# --------------------------------------------------------------------------
# Trích xuất trực tiếp từ bảng đã dựng
# --------------------------------------------------------------------------
class TestTableOrientations:
    def test_key_value_table_rows(self) -> None:
        table = table_from_rows(
            [
                ["Nội dung", "Chi tiết"],
                ["Tên mã giá OCS", "CTKMN180X"],
                ["Cước đăng ký", "163,636.3636"],
                ["Cước thuê bao tháng", "0"],
                ["Ưu đãi SMS", "100"],
                ["Chu kỳ gia hạn", "tháng"],
                ["Gói cước áp dụng", "Basic+, Family"],
            ]
        )
        payload = CTKMExtractor(use_pp_structure=False).extract_from_table(table).to_dict()
        assert payload["packageCode"] == "CTKMN180X"
        assert payload["registerFee"] == pytest.approx(163636.3636)
        assert payload["monthlyFee"] == 0
        assert payload["sms"] == 100
        assert payload["cycle"] == "tháng"
        assert payload["allowedPackages"] == ["Basic+", "Family"]

    def test_header_on_top_table(self) -> None:
        table = table_from_rows(
            [
                ["Mã gói", "Cước đăng ký", "SMS", "Data"],
                ["CTKMN180X", "163.636", "100", "60GB"],
            ]
        )
        payload = CTKMExtractor(use_pp_structure=False).extract_from_table(table).to_dict()
        assert payload["packageCode"] == "CTKMN180X"
        assert payload["registerFee"] == 163636
        assert payload["sms"] == 100
        assert payload["dataGB"] == 60

    def test_column_order_does_not_matter(self) -> None:
        table = table_from_rows(
            [
                ["SMS", "Mã gói", "Cước đăng ký"],
                ["100", "CTKMN180X", "163.636"],
            ]
        )
        payload = CTKMExtractor(use_pp_structure=False).extract_from_table(table).to_dict()
        assert payload["packageCode"] == "CTKMN180X"
        assert payload["sms"] == 100
        assert payload["registerFee"] == 163636

    def test_table_built_from_token_coordinates(self) -> None:
        """Tokens hai cột -> tầng table cluster đúng nhãn/giá trị theo toạ độ."""
        rows = [
            ("Tên mã giá OCS", "CTKMN180X"),
            ("Cước đăng ký", "163,636.3636"),
            ("Ưu đãi SMS", "100"),
        ]
        tokens: list[OCRToken] = []
        for index, (label, value) in enumerate(rows):
            top = 40.0 * index
            tokens.append(OCRToken(label, BoundingBox(10, top, 200, top + 25)))
            tokens.append(OCRToken(value, BoundingBox(320, top, 520, top + 25)))
        ocr_result = OCRResult(tokens=tokens, image_path=None, provider="fake")

        result = CTKMExtractor(use_pp_structure=False).extract_from_ocr(ocr_result)
        assert result.table is not None
        assert result.table.n_cols == 2
        assert result.table_strategy == "cluster"
        payload = result.to_dict()
        assert payload["packageCode"] == "CTKMN180X"
        assert payload["registerFee"] == pytest.approx(163636.3636)
        assert payload["sms"] == 100


# --------------------------------------------------------------------------
# Tầng table: PP-Structure và các mức fallback
# --------------------------------------------------------------------------
class TestTableLayer:
    def test_parse_table_html_handles_merged_cells(self) -> None:
        html = (
            "<table><tr><td rowspan='2'>Nội dung</td><td>Tên mã giá OCS</td></tr>"
            "<tr><td>Cước đăng ký</td></tr></table>"
        )
        assert parse_table_html(html) == [
            {"row": 0, "col": 0, "row_span": 2, "col_span": 1},
            {"row": 0, "col": 1, "row_span": 1, "col_span": 1},
            {"row": 1, "col": 1, "row_span": 1, "col_span": 1},
        ]

    def test_parse_table_html_of_garbage_returns_empty(self) -> None:
        assert parse_table_html("") == []
        assert parse_table_html("khong phai html") == []

    def test_pp_structure_cells_get_text_from_ocr_tokens(self) -> None:
        """Cấu trúc ô lấy từ PP-Structure, text lấy từ VietOCR, ghép theo toạ độ."""
        html = (
            "<table><tr><td rowspan='2'>Nội dung</td><td>Tên mã giá OCS</td>"
            "<td>CTKMN180X</td></tr>"
            "<tr><td>Ưu đãi SMS</td><td>100</td></tr></table>"
        )
        bboxes = [
            [0, 0, 100, 80],  # ô gộp dọc
            [100, 0, 300, 40],
            [300, 0, 460, 40],
            [100, 40, 300, 80],
            [300, 40, 460, 80],
        ]
        tokens = [
            OCRToken("Nội dung", BoundingBox(10, 20, 90, 45)),
            OCRToken("Tên mã giá OCS", BoundingBox(110, 5, 290, 35)),
            OCRToken("CTKMN180X", BoundingBox(310, 5, 450, 35)),
            OCRToken("Ưu đãi SMS", BoundingBox(110, 45, 290, 75)),
            OCRToken("100", BoundingBox(310, 45, 380, 75)),
        ]
        cells = PPStructureTableRecognizer._build_cells(html, bboxes)
        assigned = assign_tokens_to_cells(cells, tokens)

        assert assigned == len(tokens)
        table = Table(cells=cells, source="pp_structure", confidence=1.0)
        assert table.n_rows == 2 and table.n_cols == 3
        assert table.cell_at(1, 0) is table.cell_at(0, 0)  # ô gộp dọc

        payload = CTKMExtractor(use_pp_structure=False).extract_from_table(table).to_dict()
        assert payload["packageCode"] == "CTKMN180X"
        assert payload["sms"] == 100

    def test_build_table_uses_pp_structure_when_available(self) -> None:
        expected = table_from_rows([["Ưu đãi SMS", "100"]], source="pp_structure")

        class FakeRecognizer:
            def recognize(self, image: object, tokens: object) -> Table:
                return expected

        ocr_result = OCRResult(
            tokens=[OCRToken("Ưu đãi SMS 100", BoundingBox(0, 0, 200, 20))],
            image_path="sample.jpg",
            provider="fake",
        )
        result = build_table(ocr_result, recognizer=FakeRecognizer())  # type: ignore[arg-type]
        assert result.strategy == STRATEGY_PP_STRUCTURE
        assert result.table is expected

    def test_build_table_falls_back_to_cluster(self) -> None:
        class BrokenRecognizer:
            def recognize(self, image: object, tokens: object) -> Table:
                raise TableStructureUnavailableError("model chưa tải được")

        tokens = [
            OCRToken("Ưu đãi SMS", BoundingBox(0, 0, 150, 20)),
            OCRToken("100", BoundingBox(300, 0, 360, 20)),
            OCRToken("Chu kỳ", BoundingBox(0, 40, 150, 60)),
            OCRToken("tháng", BoundingBox(300, 40, 380, 60)),
        ]
        ocr_result = OCRResult(tokens=tokens, image_path="sample.jpg", provider="fake")

        result = build_table(ocr_result, recognizer=BrokenRecognizer())  # type: ignore[arg-type]
        assert result.strategy == STRATEGY_CLUSTER
        assert result.table is not None
        assert result.table.n_cols == 2
        assert any("PP-Structure" in warning for warning in result.warnings)

    def test_build_table_falls_back_to_raw_text(self) -> None:
        ocr_result = OCRResult(
            tokens=[OCRToken("chỉ một dòng", BoundingBox(0, 0, 100, 20))],
            image_path="sample.jpg",
            provider="fake",
        )
        result = build_table(ocr_result, use_pp_structure=False)
        assert result.strategy == STRATEGY_RAW_TEXT
        assert result.table is None
        assert result.raw_text == "chỉ một dòng"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
class TestCli:
    def test_cli_writes_json_file(self, tmp_path: Path) -> None:
        text_file = tmp_path / "raw.txt"
        text_file.write_text(SAMPLE_BLOB, encoding="utf-8")
        out_file = tmp_path / "result.json"

        exit_code = main(["--text-file", str(text_file), "--out", str(out_file)])

        assert exit_code == 0
        payload = json.loads(out_file.read_text(encoding="utf-8"))
        assert list(payload.keys()) == EXPECTED_FIELDS
        assert payload["packageCode"] == "CTKMN180X"
        assert payload["allowedPackages"] == ["Basic+", "Family", "Corporate++"]

    def test_cli_missing_image_returns_error_code(self, tmp_path: Path) -> None:
        assert main(["--image", str(tmp_path / "khong_co.jpg")]) == 1

    def test_cli_debug_prints_report(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        text_file = tmp_path / "raw.txt"
        text_file.write_text(SAMPLE_BLOB, encoding="utf-8")

        exit_code = main(["--text-file", str(text_file), "--debug"])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "OCR raw text" in captured.err
        assert "CTKMN180X" in captured.out


# --------------------------------------------------------------------------
# Cắt giá trị tại nhãn của field kế tiếp
# --------------------------------------------------------------------------
#: Dòng raw text lấy NGUYÊN VĂN từ log chạy Paddle+VietOCR trên biểu mẫu BM.12
#: thật: OCR gộp cả HÀNG HEADER của bảng vào MỘT dòng. Ô giá trị của "Cước TB"
#: bị detector bỏ sót (chữ mờ nằm dưới watermark), nên nếu không cắt phần dư thì
#: monthlyFee vớ phải "533" trong "(TK533)" - mã tài khoản của cột Data.
FLATTENED_HEADER_ROW = (
    "hưởng được Tên mã giá ocs (VNĐ/tháng) Phí đăng ký (VND tháng) Cước TB "
    "Thoại ngoại mạng Trong nước SMS Lưu lượng Data đa hướng (TK533) Ưu đãi data dơn"
)


class TestCutAtNextLabel:
    def test_cat_tai_nhan_cua_field_khac(self) -> None:
        value = cut_at_next_label(
            "Thoại ngoại mạng Trong nước SMS Lưu lượng Data đa hướng (TK533)",
            ["Cước TB"],
            ["Thoại ngoại mạng", "SMS", "Lưu lượng data"],
        )
        assert value == ""

    def test_giu_nguyen_khi_khong_gap_nhan_nao(self) -> None:
        assert cut_at_next_label("180.000 đ", ["Cước TB"], ["Thoại ngoại mạng"]) == "180.000 đ"

    def test_khong_cat_tai_alias_cua_chinh_no(self) -> None:
        # Nhãn lặp lại trong phần giá trị không được coi là mốc kết thúc.
        value = cut_at_next_label(
            "180.000 (Cước thuê bao tháng)",
            ["Cước thuê bao tháng"],
            ["Cước thuê bao tháng", "Thoại ngoại mạng"],
        )
        assert value == "180.000 (Cước thuê bao tháng)"

    def test_alias_qua_ngan_khong_lam_moc_cat(self) -> None:
        # "SMS" (3 ký tự) không được cắt cụt một giá trị hợp lệ.
        assert cut_at_next_label("100 SMS/tháng", ["Ưu đãi"], ["SMS"]) == "100 SMS/tháng"

    def test_chi_cat_tai_ranh_gioi_tu(self) -> None:
        # "Youtube" nằm lọt trong "MyYoutubeX" thì không phải một nhãn riêng.
        assert (
            cut_at_next_label("25gb MyYoutubeX", ["Ưu đãi"], ["Youtube"]) == "25gb MyYoutubeX"
        )
        assert cut_at_next_label("25gb Youtube: 30", ["Ưu đãi"], ["Youtube"]) == "25gb"


class TestFlattenedTableRow:
    """OCR gộp cả hàng bảng thành một dòng thì không được lấy số của cột khác."""

    def test_khong_lay_so_cua_cot_khac_lam_monthly_fee(self) -> None:
        table = table_from_rows(
            [
                ["Tên mã giá ocs", "Phí đăng ký (VNĐ/tháng)", "Cước TB (VND tháng)", ""],
                ["CTKMN180X", "163,636.3636", "", "MP 20p đầu tiên"],
            ]
        )

        result = CTKMExtractor().extract_from_table(table, FLATTENED_HEADER_ROW)

        # Ô giá trị vắng mặt -> null, KHÔNG được bịa ra 533 từ "(TK533)".
        assert result.to_dict()["monthlyFee"] is None

    def test_khong_nhay_xuong_dong_duoi_khi_phan_du_bi_cat(self) -> None:
        raw = FLATTENED_HEADER_ROW + "\nMP 20p đầu 60GB/tháng, tôi đa Youtube: 25gb/tháng"

        result = CTKMExtractor().extract_from_text(raw)

        # Nhãn "Cước TB" không chiếm trọn dòng, phần dư chỉ trống vì bị cắt tại
        # nhãn kế tiếp -> lấy dòng dưới là lấy nhầm nội dung cột khác (ra 20).
        assert result.to_dict()["monthlyFee"] is None

    def test_van_lay_duoc_gia_tri_o_dong_ke_tiep_khi_nhan_chiem_tron_dong(self) -> None:
        result = CTKMExtractor().extract_from_text("Cước thuê bao tháng\n180.000\n")

        assert result.to_dict()["monthlyFee"] == 180000


# --------------------------------------------------------------------------
# Gộp kết quả nhiều trang PDF
# --------------------------------------------------------------------------
class TestMergePageResults:
    """Bảng CTKM thường chỉ nằm ở MỘT trang trong cả tập hồ sơ."""

    @staticmethod
    def _page(**values: object) -> ExtractionResult:
        fields = {
            name: FieldResult(name=name, value=values.get(name), score=0.9 if name in values else 0.0,
                              source="table" if name in values else SOURCE_MISSING)
            for name in EXPECTED_FIELDS
        }
        return ExtractionResult(fields=fields)

    def test_chon_trang_nhieu_field_nhat_lam_trang_chinh(self) -> None:
        trong = self._page()
        co_bang = self._page(packageCode="CTKMN180X", offnetMinutes=150, sms=100)

        merged = merge_page_results([(1, trong), (2, co_bang), (3, trong)])

        assert merged.page == 2
        assert merged.pages_processed == 3
        assert merged.to_dict()["packageCode"] == "CTKMN180X"
        assert merged.fields["packageCode"].source_page == 2

    def test_bu_field_thieu_tu_trang_khac(self) -> None:
        trang1 = self._page(packageCode="CTKMN180X", offnetMinutes=150)
        trang2 = self._page(cycle="tháng")

        merged = merge_page_results([(1, trang1), (2, trang2)])

        assert merged.page == 1
        assert merged.to_dict()["cycle"] == "tháng"
        assert merged.fields["cycle"].source_page == 2
        assert any("trang 2" in w for w in merged.warnings)

    def test_khong_ghi_de_gia_tri_cua_trang_chinh(self) -> None:
        # Trang phụ có cùng field nhưng giá trị khác - trang chính phải thắng.
        chinh = self._page(packageCode="CTKMN180X", offnetMinutes=150, sms=100)
        phu = self._page(packageCode="SAI_BEN_TRANG_KHAC")

        merged = merge_page_results([(1, chinh), (2, phu)])

        assert merged.to_dict()["packageCode"] == "CTKMN180X"
        assert merged.fields["packageCode"].source_page == 1

    def test_moi_trang_deu_rong_van_tra_du_field_null(self) -> None:
        merged = merge_page_results([(1, self._page()), (2, self._page())])

        assert merged.pages_processed == 2
        assert list(merged.to_dict()) == EXPECTED_FIELDS
        assert all(value is None for value in merged.to_dict().values())

    def test_khong_co_trang_nao(self) -> None:
        merged = merge_page_results([])
        assert merged.pages_processed == 0

    def test_hoa_diem_thi_uu_tien_trang_so_nho(self) -> None:
        a, b = self._page(packageCode="A"), self._page(packageCode="B")
        assert merge_page_results([(5, b), (2, a)]).page == 2


# --------------------------------------------------------------------------
# Tự chọn cấu hình tiền xử lý
# --------------------------------------------------------------------------
class PreprocessAwareProvider(OCRProvider):
    """Provider giả trả text KHÁC NHAU tuỳ theo có nhị phân hoá hay không.

    Mô phỏng đúng tình huống thật: nhị phân hoá giúp engine này nhưng hại engine
    kia, nên không thể ép sẵn một bên.
    """

    name = "fake_preprocess"

    def __init__(self, when_binarized: str, when_not: str) -> None:
        self.preprocess_config = PreprocessConfig()
        self.when_binarized = when_binarized
        self.when_not = when_not
        self.seen: list[bool] = []

    def extract(self, image_path: str) -> OCRResult:
        binarized = self.preprocess_config.adaptive_threshold
        self.seen.append(binarized)
        text = self.when_binarized if binarized else self.when_not
        tokens = [
            OCRToken(text=line, box=BoundingBox(0, 30 * i, 400, 30 * i + 25))
            for i, line in enumerate(text.splitlines())
            if line.strip()
        ]
        return OCRResult(tokens=tokens, image_path=image_path, provider=self.name)


class TestAutoBinarize:
    ÍT = "Tên mã giá OCS: GOI_A"
    NHIỀU = "Tên mã giá OCS: GOI_B\nƯu đãi SMS: 100\nChu kỳ: tháng"

    def test_mac_dinh_chay_ca_hai_va_giu_ket_qua_tot_hon(self) -> None:
        provider = PreprocessAwareProvider(when_binarized=self.ÍT, when_not=self.NHIỀU)

        result = CTKMExtractor(provider=provider, use_pp_structure=False).extract_from_image("a.png")

        assert provider.seen == [True, False]          # đã thử cả hai
        assert result.to_dict()["packageCode"] == "GOI_B"
        assert result.to_dict()["sms"] == 100

    def test_giu_ket_qua_tot_hon_ke_ca_khi_no_o_luot_dau(self) -> None:
        provider = PreprocessAwareProvider(when_binarized=self.NHIỀU, when_not=self.ÍT)

        result = CTKMExtractor(provider=provider, use_pp_structure=False).extract_from_image("a.png")

        assert result.to_dict()["packageCode"] == "GOI_B"

    @pytest.mark.parametrize("binarize", [True, False])
    def test_ep_thi_chi_chay_mot_luot(self, binarize: bool) -> None:
        provider = PreprocessAwareProvider(when_binarized=self.ÍT, when_not=self.NHIỀU)

        CTKMExtractor(
            provider=provider, use_pp_structure=False, binarize=binarize
        ).extract_from_image("a.png")

        assert provider.seen == [binarize]

    def test_provider_khong_co_tien_xu_ly_thi_chi_chay_mot_luot(self) -> None:
        # Provider mock không có preprocess_config -> hai lượt là vô nghĩa.
        provider = FakeOCRProvider(SAMPLE_BLOB)

        CTKMExtractor(provider=provider, use_pp_structure=False).extract_from_image("x.png")

        assert provider.calls == ["x.png"]
