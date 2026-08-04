// Port của ``ctkm_extractor/tests/test_field_parsers.py``.
//
// Test từng parser với input text giả lập (không cần ảnh thật), gồm cả dữ liệu
// nhiễu/thiếu để chứng minh parser không crash và trả null hợp lý. Các giá trị
// lấy từ ảnh mẫu chỉ xuất hiện ở đây (test fixture), không nằm trong logic.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "extraction/FieldParsers.hpp"
#include "util/Json.hpp"

using ctkm::extraction::FieldValue;
using ctkm::extraction::ParserArgs;
using Catch::Approx;

namespace {

/// Dựng ParserArgs từ chuỗi JSON cho gọn.
ParserArgs args(const std::string& json) {
    return ParserArgs(ctkm::json::Value::parse(json));
}

/// Giá trị số của kết quả parse (int hoặc double đều quy về double).
double numberOf(const std::optional<FieldValue>& value) {
    REQUIRE(value.has_value());
    if (std::holds_alternative<long long>(*value)) {
        return static_cast<double>(std::get<long long>(*value));
    }
    REQUIRE(std::holds_alternative<double>(*value));
    return std::get<double>(*value);
}

std::string stringOf(const std::optional<FieldValue>& value) {
    REQUIRE(value.has_value());
    REQUIRE(std::holds_alternative<std::string>(*value));
    return std::get<std::string>(*value);
}

std::vector<std::string> listOf(const std::optional<FieldValue>& value) {
    REQUIRE(value.has_value());
    REQUIRE(std::holds_alternative<std::vector<std::string>>(*value));
    return std::get<std::vector<std::string>>(*value);
}

bool isInteger(const std::optional<FieldValue>& value) {
    return value.has_value() && std::holds_alternative<long long>(*value);
}

}  // namespace

// ---------------------------------------------------------------------------
// Tiện ích chuẩn hoá
// ---------------------------------------------------------------------------
TEST_CASE("fold_accents bỏ dấu và giữ ánh xạ vị trí", "[helpers]") {
    const std::string source = "Cước đăng ký gói CTKM tháng";
    const ctkm::extraction::FoldedText folded = ctkm::extraction::foldAccents(source);
    REQUIRE(folded.folded == "Cuoc dang ky goi CTKM thang");
    // Ánh xạ ngược phải trỏ đúng về vị trí byte trong chuỗi gốc.
    const std::size_t position = folded.folded.find("thang");
    REQUIRE(source.substr(folded.originalOffset(position), std::string("tháng").size()) ==
            "tháng");
}

TEST_CASE("fold_accents xử lý chuỗi rỗng", "[helpers]") {
    REQUIRE(ctkm::extraction::foldAccentsSimple("").empty());
}

TEST_CASE("normalize_label", "[helpers]") {
    REQUIRE(ctkm::extraction::normalizeLabel("Tên mã giá OCS") == "ten ma gia ocs");
    REQUIRE(ctkm::extraction::normalizeLabel("  Cước ĐK  (chưa VAT) ") == "cuoc dk chua vat");
    REQUIRE(ctkm::extraction::normalizeLabel("Basic+") == "basic+");
    REQUIRE(ctkm::extraction::normalizeLabel("").empty());
}

TEST_CASE("fix_ocr_digits chỉ đụng tới ngữ cảnh số", "[helpers]") {
    REQUIRE(ctkm::extraction::fixOcrDigits("1O0") == "100");
    REQUIRE(ctkm::extraction::fixOcrDigits("l50") == "150");
    REQUIRE(ctkm::extraction::fixOcrDigits("15O.OOO") == "150.000");
    // Chữ "sms" đứng tách khỏi số thì không bị đụng tới.
    REQUIRE(ctkm::extraction::fixOcrDigits("150 sms") == "150 sms");
    REQUIRE(ctkm::extraction::fixOcrDigits("").empty());
}

TEST_CASE("fix_ocr_digits giữ nguyên đường kẻ ô ở rìa cụm", "[helpers]") {
    // Đường kẻ ô dính vào cuối số: đổi thành "1" là bịa thêm một chữ số.
    REQUIRE(ctkm::extraction::fixOcrDigits("163,636.3636|") == "163,636.3636|");
    REQUIRE(ctkm::extraction::fixOcrDigits("|163,636.3636") == "|163,636.3636");
    REQUIRE(ctkm::extraction::fixOcrDigits("100|") == "100|");
    REQUIRE(ctkm::extraction::fixOcrDigits("|") == "|");
    // Nằm GIỮA hai chữ số thì vẫn là chữ số bị đọc nhầm.
    REQUIRE(ctkm::extraction::fixOcrDigits("1|0") == "110");
    REQUIRE(ctkm::extraction::fixOcrDigits("15!.000") == "151.000");
}

TEST_CASE("money_parser bỏ qua đường kẻ ô dính vào số", "[money]") {
    const ParserArgs empty = args("{}");
    // Ô bảng thật của biểu mẫu BM.12 sau khi OCR: giá trị dính đường kẻ.
    REQUIRE(numberOf(ctkm::extraction::moneyParser("163,636.3636|", empty)) ==
            Approx(163636.3636));
    REQUIRE(numberOf(ctkm::extraction::moneyParser("| 150.534.213 |", empty)) ==
            Approx(150534213));
}

TEST_CASE("slice_after_keyword không phân biệt dấu", "[helpers]") {
    REQUIRE(ctkm::extraction::trim(
                ctkm::extraction::sliceAfterKeyword("Chu kỳ gia hạn: tháng", "chu ky")) ==
            "gia hạn: tháng");
    // Không thấy keyword thì giữ nguyên chuỗi.
    REQUIRE(ctkm::extraction::sliceAfterKeyword("60GB/tháng", "Youtube") == "60GB/tháng");
}

// ---------------------------------------------------------------------------
// money_parser
// ---------------------------------------------------------------------------
TEST_CASE("money_parser xử lý dấu phân cách lẫn lộn", "[money]") {
    const ParserArgs empty = args("{}");

    // Lẫn lộn "," và "." -> dấu cuối cùng là thập phân.
    REQUIRE(numberOf(ctkm::extraction::moneyParser("163,636.3636", empty)) ==
            Approx(163636.3636));
    REQUIRE(numberOf(ctkm::extraction::moneyParser("163.636,3636", empty)) ==
            Approx(163636.3636));
    // Cùng một loại dấu, mọi nhóm 3 chữ số -> tất cả là hàng nghìn.
    REQUIRE(numberOf(ctkm::extraction::moneyParser("150.534.213", empty)) ==
            Approx(150534213.0));
    REQUIRE(numberOf(ctkm::extraction::moneyParser("150,534,213", empty)) ==
            Approx(150534213.0));
    // Một dấu duy nhất.
    REQUIRE(numberOf(ctkm::extraction::moneyParser("150.000", empty)) == Approx(150000.0));
    REQUIRE(numberOf(ctkm::extraction::moneyParser("163,6363", empty)) == Approx(163.6363));
    REQUIRE(numberOf(ctkm::extraction::moneyParser("1.5", empty)) == Approx(1.5));
    // Có đơn vị / khoảng trắng / nhiễu xung quanh.
    REQUIRE(numberOf(ctkm::extraction::moneyParser("163,636.3636 đ", empty)) ==
            Approx(163636.3636));
    REQUIRE(numberOf(ctkm::extraction::moneyParser("Cước ĐK: 163 636", empty)) ==
            Approx(163636.0));
    REQUIRE(numberOf(ctkm::extraction::moneyParser("0", empty)) == Approx(0.0));
    REQUIRE(numberOf(ctkm::extraction::moneyParser("0 đồng", empty)) == Approx(0.0));
}

TEST_CASE("money_parser trả số nguyên khi không có phần thập phân", "[money]") {
    REQUIRE(isInteger(ctkm::extraction::moneyParser("150.000", args("{}"))));
    REQUIRE_FALSE(isInteger(ctkm::extraction::moneyParser("163,636.3636", args("{}"))));
}

TEST_CASE("money_parser sửa được nhiễu OCR trong cụm số", "[money]") {
    REQUIRE(numberOf(ctkm::extraction::moneyParser("l63,636.3636", args("{}"))) ==
            Approx(163636.3636));
    REQUIRE(numberOf(ctkm::extraction::moneyParser("15O.OOO", args("{}"))) == Approx(150000.0));
}

TEST_CASE("money_parser trả null với dữ liệu thiếu/nhiễu", "[money]") {
    const ParserArgs empty = args("{}");
    for (const std::string& raw : {"", "Miễn phí", "N/A", "---", "abc"}) {
        REQUIRE_FALSE(ctkm::extraction::moneyParser(raw, empty).has_value());
    }
}

TEST_CASE("money_parser kiểm tra ngưỡng min/max", "[money]") {
    REQUIRE_FALSE(ctkm::extraction::moneyParser("-500", args(R"({"min_value": 0})")).has_value());
    REQUIRE_FALSE(
        ctkm::extraction::moneyParser("999999999", args(R"({"max_value": 1000})")).has_value());
}

TEST_CASE("money_parser thu hẹp phạm vi theo keyword", "[money]") {
    const std::string raw = "Cước đăng ký 163,636.3636 - Cước tháng 0";
    REQUIRE(numberOf(ctkm::extraction::moneyParser(raw, args(R"({"keyword": "Cước tháng"})"))) ==
            Approx(0.0));
}

TEST_CASE("money_parser lấy được cụm số thứ hai", "[money]") {
    REQUIRE(numberOf(ctkm::extraction::moneyParser("60 GB, tối đa 8 GB",
                                                   args(R"({"occurrence": 1})"))) == Approx(8.0));
}

// ---------------------------------------------------------------------------
// int_parser
// ---------------------------------------------------------------------------
TEST_CASE("int_parser lấy số nguyên đầu tiên", "[int]") {
    const ParserArgs empty = args("{}");
    REQUIRE(numberOf(ctkm::extraction::intParser("150", empty)) == Approx(150.0));
    REQUIRE(numberOf(ctkm::extraction::intParser("150 phút", empty)) == Approx(150.0));
    REQUIRE(numberOf(ctkm::extraction::intParser("100", empty)) == Approx(100.0));
    REQUIRE(numberOf(ctkm::extraction::intParser("MP 20p đầu tiên", empty)) == Approx(20.0));
    REQUIRE(numberOf(ctkm::extraction::intParser("1.500 phút ngoại mạng", empty)) ==
            Approx(1500.0));
    REQUIRE(numberOf(ctkm::extraction::intParser("Miễn phí 20 phút đầu", empty)) ==
            Approx(20.0));
}

TEST_CASE("int_parser làm tròn số thập phân", "[int]") {
    REQUIRE(numberOf(ctkm::extraction::intParser("149,6", args("{}"))) == Approx(150.0));
}

TEST_CASE("int_parser trả null với dữ liệu thiếu", "[int]") {
    const ParserArgs empty = args("{}");
    for (const std::string& raw : {"", "không", "-", "phút"}) {
        REQUIRE_FALSE(ctkm::extraction::intParser(raw, empty).has_value());
    }
}

TEST_CASE("int_parser kiểm tra ngưỡng", "[int]") {
    REQUIRE_FALSE(ctkm::extraction::intParser("-5", args(R"({"min_value": 0})")).has_value());
    REQUIRE_FALSE(
        ctkm::extraction::intParser("100000", args(R"({"max_value": 10000})")).has_value());
}

// ---------------------------------------------------------------------------
// gb_parser
// ---------------------------------------------------------------------------
TEST_CASE("gb_parser lấy dung lượng GB đầu tiên", "[gb]") {
    const ParserArgs empty = args("{}");
    REQUIRE(numberOf(ctkm::extraction::gbParser("60GB/tháng, tối đa 8gb/1 ngày", empty)) ==
            Approx(60.0));
    REQUIRE(numberOf(ctkm::extraction::gbParser("60 GB", empty)) == Approx(60.0));
    REQUIRE(numberOf(ctkm::extraction::gbParser("Data: 30gb", empty)) == Approx(30.0));
    REQUIRE(numberOf(ctkm::extraction::gbParser("1.5GB/ngày", empty)) == Approx(1.5));
    REQUIRE(numberOf(ctkm::extraction::gbParser("60 G B", empty)) == Approx(60.0));
}

TEST_CASE("gb_parser thu hẹp phạm vi theo keyword", "[gb]") {
    const std::string raw = "Youtube: 25gb/tháng; Spotify: 10gb/tháng";
    REQUIRE(numberOf(ctkm::extraction::gbParser(raw, args(R"({"keyword": "Youtube"})"))) ==
            Approx(25.0));
    REQUIRE(numberOf(ctkm::extraction::gbParser(raw, args(R"({"keyword": "Spotify"})"))) ==
            Approx(10.0));
}

TEST_CASE("gb_parser không thấy keyword thì dò trên toàn chuỗi", "[gb]") {
    // Ô bảng đã giới hạn phạm vi rồi nên vẫn lấy được số.
    REQUIRE(numberOf(ctkm::extraction::gbParser("25gb/tháng",
                                                args(R"({"keyword": "Youtube"})"))) ==
            Approx(25.0));
}

TEST_CASE("gb_parser với require_keyword trả null khi thiếu keyword", "[gb]") {
    REQUIRE_FALSE(
        ctkm::extraction::gbParser("25gb/tháng",
                                   args(R"({"keyword": "Spotify", "require_keyword": true})"))
            .has_value());
}

TEST_CASE("gb_parser fallback về số thuần", "[gb]") {
    REQUIRE(numberOf(ctkm::extraction::gbParser("60", args("{}"))) == Approx(60.0));
    REQUIRE_FALSE(
        ctkm::extraction::gbParser("60", args(R"({"fallback_to_number": false})")).has_value());
}

TEST_CASE("gb_parser trả null với dữ liệu thiếu", "[gb]") {
    const ParserArgs empty = args("{}");
    for (const std::string& raw : {"", "Không giới hạn", "GB"}) {
        REQUIRE_FALSE(ctkm::extraction::gbParser(raw, empty).has_value());
    }
}

// ---------------------------------------------------------------------------
// text_parser - dùng cho onnetMinutes và packageCode
// ---------------------------------------------------------------------------
TEST_CASE("text_parser làm sạch và viết hoa mã gói", "[text]") {
    const auto value = ctkm::extraction::textParser(
        " : ctkmn180x ", args(R"({"pattern": "[A-Za-z0-9][A-Za-z0-9._+-]{2,}", "upper": true})"));
    REQUIRE(stringOf(value) == "CTKMN180X");
}

TEST_CASE("text_parser lấy mã từ ô nhiễu", "[text]") {
    const auto value = ctkm::extraction::textParser(
        "Tên mã giá OCS CTKMN180X", args(R"({"pattern": "[A-Z][A-Z0-9]{4,}", "upper": true})"));
    REQUIRE(stringOf(value) == "CTKMN180X");
}

TEST_CASE("text_parser giữ nguyên mô tả chính sách của onnetMinutes", "[text]") {
    // LƯU Ý: onnetMinutes là mô tả ("MP 20p đầu tiên"), KHÔNG ép về số.
    const auto value =
        ctkm::extraction::textParser("MP 20p đầu tiên", args(R"({"max_length": 80})"));
    REQUIRE(stringOf(value) == "MP 20p đầu tiên");
}

TEST_CASE("text_parser cắt theo số KÝ TỰ chứ không phải byte", "[text]") {
    // "MP 20p đầu tiên" có 15 ký tự nhưng nhiều hơn 15 byte (UTF-8).
    const auto value =
        ctkm::extraction::textParser("MP 20p đầu tiên", args(R"({"max_length": 6})"));
    REQUIRE(stringOf(value) == "MP 20p");
}

TEST_CASE("text_parser trả null khi pattern không khớp", "[text]") {
    REQUIRE_FALSE(
        ctkm::extraction::textParser("---", args(R"({"pattern": "[A-Z0-9]{4,}"})")).has_value());
}

TEST_CASE("text_parser không crash với pattern hỏng", "[text]") {
    REQUIRE_FALSE(
        ctkm::extraction::textParser("CTKMN180X", args(R"({"pattern": "[unclosed"})"))
            .has_value());
}

TEST_CASE("text_parser trả null với dữ liệu thiếu", "[text]") {
    const ParserArgs empty = args("{}");
    for (const std::string& raw : {"", "   ", ":"}) {
        REQUIRE_FALSE(ctkm::extraction::textParser(raw, empty).has_value());
    }
}

// ---------------------------------------------------------------------------
// cycle_parser
// ---------------------------------------------------------------------------
TEST_CASE("cycle_parser trích từ khoá chu kỳ", "[cycle]") {
    const ParserArgs empty = args("{}");
    REQUIRE(stringOf(ctkm::extraction::cycleParser("Chu kỳ gia hạn: tháng", empty)) == "tháng");
    REQUIRE(stringOf(ctkm::extraction::cycleParser("tháng", empty)) == "tháng");
    REQUIRE(stringOf(ctkm::extraction::cycleParser("Chu kỳ: 30 ngày", empty)) == "ngày");
    REQUIRE(stringOf(ctkm::extraction::cycleParser("Gia hạn theo tuần", empty)) == "tuần");
    // Mất dấu do OCR vẫn nhận ra được.
    REQUIRE(stringOf(ctkm::extraction::cycleParser("Chu ky gia han: thang", empty)) == "tháng");
}

TEST_CASE("cycle_parser với danh sách từ khoá tuỳ biến", "[cycle]") {
    REQUIRE(stringOf(ctkm::extraction::cycleParser(
                "Chu kỳ: quý", args(R"({"keywords": ["quý", "năm"]})"))) == "quý");
}

TEST_CASE("cycle_parser trả null với dữ liệu thiếu", "[cycle]") {
    const ParserArgs empty = args("{}");
    for (const std::string& raw : {"", "Chu kỳ: ???", "12345"}) {
        REQUIRE_FALSE(ctkm::extraction::cycleParser(raw, empty).has_value());
    }
}

// ---------------------------------------------------------------------------
// list_parser
// ---------------------------------------------------------------------------
TEST_CASE("list_parser tách theo dấu phẩy và giữ dấu cộng", "[list]") {
    const auto items = listOf(ctkm::extraction::listParser("Basic+, Family, Corporate++",
                                                           args("{}")));
    REQUIRE(items == std::vector<std::string>{"Basic+", "Family", "Corporate++"});
}

TEST_CASE("list_parser xử lý separator và khoảng trắng thừa", "[list]") {
    const auto items = listOf(
        ctkm::extraction::listParser(" Basic+ ;  Family \n Corporate++ . ", args("{}")));
    REQUIRE(items == std::vector<std::string>{"Basic+", "Family", "Corporate++"});
}

TEST_CASE("list_parser loại trùng lặp và phần tử quá ngắn", "[list]") {
    const auto items = listOf(ctkm::extraction::listParser(
        "Basic+, Basic+, x, Family", args(R"({"min_item_length": 2})")));
    REQUIRE(items == std::vector<std::string>{"Basic+", "Family"});
}

TEST_CASE("list_parser bỏ tiền tố nhãn theo keyword", "[list]") {
    const auto items = listOf(ctkm::extraction::listParser(
        "Gói cước được phép ĐK: Basic+, Family",
        args(R"({"keyword": "Gói cước được phép ĐK"})")));
    REQUIRE(items == std::vector<std::string>{"Basic+", "Family"});
}

TEST_CASE("list_parser giới hạn số phần tử", "[list]") {
    const auto items =
        listOf(ctkm::extraction::listParser("A1, B2, C3", args(R"({"max_items": 2})")));
    REQUIRE(items == std::vector<std::string>{"A1", "B2"});
}

TEST_CASE("list_parser trả null với dữ liệu thiếu", "[list]") {
    const ParserArgs empty = args("{}");
    for (const std::string& raw : {"", "   ", ",,,", "-"}) {
        REQUIRE_FALSE(ctkm::extraction::listParser(raw, empty).has_value());
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------
TEST_CASE("registry có đủ parser dùng trong schema", "[registry]") {
    for (const std::string& name : {"money_parser", "int_parser", "gb_parser", "cycle_parser",
                                    "list_parser", "text_parser"}) {
        REQUIRE(ctkm::extraction::parserRegistry().count(name) == 1);
    }
}

TEST_CASE("parser không tồn tại fallback về text_parser", "[registry]") {
    const auto value =
        ctkm::extraction::parseValue("khong_ton_tai", "CTKMN180X", args("{}"));
    REQUIRE(stringOf(value) == "CTKMN180X");
}

TEST_CASE("parse_value không bao giờ ném exception", "[registry]") {
    // Pattern hỏng khiến std::regex ném lỗi bên trong parser.
    REQUIRE_NOTHROW(ctkm::extraction::parseValue("text_parser", "abc",
                                                 args(R"({"pattern": "([unclosed"})")));
    REQUIRE_NOTHROW(ctkm::extraction::parseValue("gb_parser", "60GB",
                                                 args(R"({"unit_pattern": "([unclosed"})")));
}

TEST_CASE("toJson trả null cho giá trị thiếu", "[registry]") {
    REQUIRE(ctkm::extraction::toJson(std::nullopt).dump() == "null");
    REQUIRE(ctkm::extraction::toJson(FieldValue{static_cast<long long>(0)}).dump() == "0");
    REQUIRE(ctkm::extraction::toJson(FieldValue{std::string("tháng")}).dump() == "\"tháng\"");
}
