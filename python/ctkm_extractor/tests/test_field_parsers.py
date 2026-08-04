"""Test từng parser với input text giả lập (không cần ảnh thật).

Bao gồm cả dữ liệu nhiễu/thiếu để chứng minh parser không crash và trả ``None``
hợp lý. Các giá trị lấy từ ảnh mẫu chỉ xuất hiện ở đây (test fixture), không nằm
trong logic trích xuất.
"""

from __future__ import annotations

import pytest

from ctkm_extractor.extraction.field_parsers import (
    PARSERS,
    available_parsers,
    coerce_text,
    cycle_parser,
    fix_ocr_digits,
    fold_accents,
    gb_parser,
    get_parser,
    int_parser,
    list_parser,
    money_parser,
    normalize_label,
    parse_many,
    parse_value,
    raw_parser,
    register,
    slice_after_keyword,
    text_parser,
)


# ---------------------------------------------------------------------------
# Tiện ích chuẩn hoá
# ---------------------------------------------------------------------------
class TestNormalizationHelpers:
    def test_fold_accents_preserves_length(self) -> None:
        source = "Cước đăng ký gói CTKM tháng"
        folded = fold_accents(source)
        assert folded == "Cuoc dang ky goi CTKM thang"
        assert len(folded) == len(source)

    def test_fold_accents_handles_empty_and_none(self) -> None:
        assert fold_accents("") == ""
        assert fold_accents(None) == ""  # type: ignore[arg-type]

    def test_normalize_label(self) -> None:
        assert normalize_label("Tên mã giá OCS") == "ten ma gia ocs"
        assert normalize_label("  Cước ĐK  (chưa VAT) ") == "cuoc dk chua vat"
        assert normalize_label("Basic+") == "basic+"
        assert normalize_label("") == ""

    def test_coerce_text(self) -> None:
        assert coerce_text(None) == ""
        assert coerce_text(" 150 ") == "150"
        assert coerce_text(["Basic+", "Family"]) == "Basic+ Family"
        assert coerce_text(120) == "120"

    def test_fix_ocr_digits_only_touches_digit_neighbours(self) -> None:
        assert fix_ocr_digits("1O0") == "100"
        assert fix_ocr_digits("l50") == "150"
        # Chữ "sms" đứng tách khỏi số thì không bị đụng tới.
        assert fix_ocr_digits("150 sms") == "150 sms"
        assert fix_ocr_digits("") == ""

    def test_fix_ocr_digits_keeps_table_border_at_chunk_edge(self) -> None:
        # Đường kẻ ô dính vào cuối số: đổi thành "1" là bịa thêm một chữ số.
        assert fix_ocr_digits("163,636.3636|") == "163,636.3636|"
        assert fix_ocr_digits("|163,636.3636") == "|163,636.3636"
        assert fix_ocr_digits("100|") == "100|"
        assert fix_ocr_digits("|") == "|"
        # Nằm GIỮA hai chữ số thì vẫn là chữ số bị đọc nhầm.
        assert fix_ocr_digits("1|0") == "110"
        assert fix_ocr_digits("15!.000") == "151.000"

    def test_money_parser_ignores_trailing_table_border(self) -> None:
        # Ô bảng thật của biểu mẫu BM.12 sau khi OCR: giá trị dính đường kẻ.
        assert money_parser("163,636.3636|") == 163636.3636
        assert money_parser("| 150.534.213 |") == 150534213

    def test_slice_after_keyword_is_accent_insensitive(self) -> None:
        assert slice_after_keyword("Chu kỳ gia hạn: tháng", "chu ky").strip() == "gia hạn: tháng"
        assert slice_after_keyword("Chu kỳ gia hạn: tháng", "chu ky gia han").strip() == ": tháng"
        # Không thấy keyword thì giữ nguyên chuỗi.
        assert slice_after_keyword("60GB/tháng", "Youtube") == "60GB/tháng"


# ---------------------------------------------------------------------------
# money_parser
# ---------------------------------------------------------------------------
class TestMoneyParser:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            # Lẫn lộn "," và "." -> dấu cuối cùng là thập phân.
            ("163,636.3636", 163636.3636),
            ("163.636,3636", 163636.3636),
            # Cùng một loại dấu, mọi nhóm 3 chữ số -> tất cả là hàng nghìn.
            ("150.534.213", 150534213),
            ("150,534,213", 150534213),
            # Một dấu duy nhất.
            ("150.000", 150000),
            ("163,6363", 163.6363),
            ("1.5", 1.5),
            # Có ký tự đơn vị / khoảng trắng / nhiễu xung quanh.
            ("163,636.3636 đ", 163636.3636),
            ("Cước ĐK: 163 636", 163636),
            ("0", 0),
            ("0 đồng", 0),
        ],
    )
    def test_parses_mixed_separators(self, raw: str, expected: float) -> None:
        assert money_parser(raw) == pytest.approx(expected)

    def test_returns_int_when_no_fraction(self) -> None:
        value = money_parser("150.000")
        assert isinstance(value, int)

    def test_ocr_noise_is_repaired(self) -> None:
        # "O" bị OCR nhầm thay cho số 0.
        assert money_parser("l63,636.3636") == pytest.approx(163636.3636)
        assert money_parser("15O.OOO") == 150000

    @pytest.mark.parametrize("raw", ["", None, "Miễn phí", "N/A", "---", "abc"])
    def test_missing_or_noisy_input_returns_none(self, raw: object) -> None:
        assert money_parser(raw) is None

    def test_value_out_of_range_returns_none(self) -> None:
        assert money_parser("-500", min_value=0) is None
        assert money_parser("999999999", max_value=1000) is None

    def test_keyword_narrows_scope(self) -> None:
        raw = "Cước đăng ký 163,636.3636 - Cước tháng 0"
        assert money_parser(raw, keyword="Cước tháng") == 0

    def test_occurrence_selects_later_number(self) -> None:
        assert money_parser("60 GB, tối đa 8 GB", occurrence=1) == 8


# ---------------------------------------------------------------------------
# int_parser
# ---------------------------------------------------------------------------
class TestIntParser:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("150", 150),
            ("150 phút", 150),
            ("100", 100),
            ("MP 20p đầu tiên", 20),
            ("1.500 phút ngoại mạng", 1500),
            ("Miễn phí 20 phút đầu", 20),
        ],
    )
    def test_first_integer_is_returned(self, raw: str, expected: int) -> None:
        assert int_parser(raw) == expected

    def test_rounds_decimal_values(self) -> None:
        assert int_parser("149,6") == 150

    @pytest.mark.parametrize("raw", ["", None, "không", "-", "phút"])
    def test_missing_input_returns_none(self, raw: object) -> None:
        assert int_parser(raw) is None

    def test_range_guard(self) -> None:
        assert int_parser("-5", min_value=0) is None
        assert int_parser("100000", max_value=10000) is None


# ---------------------------------------------------------------------------
# gb_parser
# ---------------------------------------------------------------------------
class TestGbParser:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("60GB/tháng, tối đa 8gb/1 ngày", 60),
            ("60 GB", 60),
            ("Data: 30gb", 30),
            ("1.5GB/ngày", 1.5),
            ("60 G B", 60),
        ],
    )
    def test_reads_first_gb_value(self, raw: str, expected: float) -> None:
        assert gb_parser(raw) == pytest.approx(expected)

    def test_keyword_narrows_scope(self) -> None:
        raw = "Youtube: 25gb/tháng; Spotify: 10gb/tháng"
        assert gb_parser(raw, keyword="Youtube") == 25
        assert gb_parser(raw, keyword="Spotify") == 10

    def test_keyword_missing_falls_back_to_whole_string(self) -> None:
        # Ô bảng đã giới hạn phạm vi rồi nên vẫn lấy được số.
        assert gb_parser("25gb/tháng", keyword="Youtube") == 25

    def test_require_keyword_returns_none_when_absent(self) -> None:
        assert gb_parser("25gb/tháng", keyword="Spotify", require_keyword=True) is None

    def test_plain_number_uses_fallback(self) -> None:
        assert gb_parser("60") == 60
        assert gb_parser("60", fallback_to_number=False) is None

    @pytest.mark.parametrize("raw", ["", None, "Không giới hạn", "GB"])
    def test_missing_input_returns_none(self, raw: object) -> None:
        assert gb_parser(raw) is None


# ---------------------------------------------------------------------------
# cycle_parser
# ---------------------------------------------------------------------------
class TestCycleParser:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Chu kỳ gia hạn: tháng", "tháng"),
            ("tháng", "tháng"),
            ("Chu kỳ: 30 ngày", "ngày"),
            ("Gia hạn theo tuần", "tuần"),
            ("CHU KỲ GIA HẠN: THÁNG", "tháng"),
            # Mất dấu do OCR vẫn nhận ra được.
            ("Chu ky gia han: thang", "tháng"),
        ],
    )
    def test_extracts_cycle_keyword(self, raw: str, expected: str) -> None:
        assert cycle_parser(raw) == expected

    def test_custom_keywords(self) -> None:
        assert cycle_parser("Chu kỳ: quý", keywords=["quý", "năm"]) == "quý"

    @pytest.mark.parametrize("raw", ["", None, "Chu kỳ: ???", "12345"])
    def test_missing_input_returns_none(self, raw: object) -> None:
        assert cycle_parser(raw) is None


# ---------------------------------------------------------------------------
# list_parser
# ---------------------------------------------------------------------------
class TestListParser:
    def test_splits_by_comma_and_keeps_plus_signs(self) -> None:
        assert list_parser("Basic+, Family, Corporate++") == [
            "Basic+",
            "Family",
            "Corporate++",
        ]

    def test_handles_extra_separators_and_whitespace(self) -> None:
        assert list_parser(" Basic+ ;  Family \n Corporate++ . ") == [
            "Basic+",
            "Family",
            "Corporate++",
        ]

    def test_drops_duplicates_and_short_noise(self) -> None:
        assert list_parser("Basic+, Basic+, x, Family", min_item_length=2) == [
            "Basic+",
            "Family",
        ]

    def test_keyword_removes_label_prefix(self) -> None:
        raw = "Gói cước được phép ĐK: Basic+, Family"
        assert list_parser(raw, keyword="Gói cước được phép ĐK") == ["Basic+", "Family"]

    def test_max_items(self) -> None:
        assert list_parser("A1, B2, C3", max_items=2) == ["A1", "B2"]

    @pytest.mark.parametrize("raw", ["", None, "   ", ",,,", "-"])
    def test_missing_input_returns_none(self, raw: object) -> None:
        assert list_parser(raw) is None


# ---------------------------------------------------------------------------
# text_parser / raw_parser
# ---------------------------------------------------------------------------
class TestTextParser:
    def test_cleans_and_uppercases_code(self) -> None:
        value = text_parser(
            " : ctkmn180x ", pattern="[A-Za-z0-9][A-Za-z0-9._+-]{2,}", upper=True
        )
        assert value == "CTKMN180X"

    def test_pattern_picks_code_from_noisy_cell(self) -> None:
        value = text_parser(
            "Tên mã giá OCS CTKMN180X", pattern="[A-Z][A-Z0-9]{4,}", upper=True
        )
        assert value == "CTKMN180X"

    def test_pattern_not_matching_returns_none(self) -> None:
        assert text_parser("---", pattern="[A-Z0-9]{4,}") is None

    def test_invalid_pattern_does_not_crash(self) -> None:
        assert text_parser("CTKMN180X", pattern="[unclosed") is None

    def test_max_length_truncates(self) -> None:
        assert text_parser("ABCDEFGH", max_length=3) == "ABC"

    @pytest.mark.parametrize("raw", ["", None, "   ", ":"])
    def test_missing_input_returns_none(self, raw: object) -> None:
        assert text_parser(raw) is None

    def test_raw_parser_keeps_content(self) -> None:
        assert raw_parser("  60GB/tháng  ") == "60GB/tháng"
        assert raw_parser("") is None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_all_schema_parsers_are_registered(self) -> None:
        for name in (
            "money_parser",
            "int_parser",
            "gb_parser",
            "cycle_parser",
            "list_parser",
            "text_parser",
        ):
            assert name in PARSERS
            assert name in available_parsers()

    def test_unknown_parser_falls_back_to_text_parser(self) -> None:
        assert get_parser("khong_ton_tai") is PARSERS["text_parser"]

    def test_parse_value_never_raises(self) -> None:
        # Tham số sai kiểu -> parser ném TypeError, nhưng parse_value phải nuốt lỗi.
        assert parse_value("int_parser", "150", khong_ton_tai_kwarg=True) is None
        assert parse_value("money_parser", object()) is None

    def test_custom_parser_can_be_registered(self) -> None:
        @register("test_double_parser")
        def _double(raw: object) -> int | None:
            value = int_parser(raw)
            return None if value is None else value * 2

        try:
            assert parse_value("test_double_parser", "21") == 42
        finally:
            PARSERS.pop("test_double_parser", None)

    def test_parse_many(self) -> None:
        result = parse_many(
            [
                ("sms", "int_parser", "100", {}),
                ("dataGB", "gb_parser", "60GB/tháng", {}),
                ("cycle", "cycle_parser", "", {}),
            ]
        )
        assert result == {"sms": 100, "dataGB": 60, "cycle": None}
