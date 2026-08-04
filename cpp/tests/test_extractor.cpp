// Port của ``ctkm_extractor/tests/test_extractor.py``.
//
// Test orchestrator end-to-end với OCR được mock bằng raw-text blob: không cần
// ảnh thật, không cần model ONNX. Bao gồm test dùng ĐÚNG NGUYÊN VĂN header của
// ảnh mẫu thật (BM.12) - đây là cách duy nhất từng phát hiện ra các bug thực tế
// ở bản Python.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <vector>

#include "extraction/Extractor.hpp"
#include "extraction/Schema.hpp"
#include "ocr/OCRProvider.hpp"
#include "table/Reconstruct.hpp"
#include "table/TableBuilder.hpp"

using Catch::Approx;
using ctkm::extraction::CTKMExtractor;
using ctkm::extraction::ExtractionResult;
using ctkm::extraction::FieldValue;
using ctkm::ocr::BoundingBox;
using ctkm::ocr::OCRResult;
using ctkm::ocr::OCRToken;

namespace {

#ifndef CTKM_TEST_SCHEMA_PATH
#define CTKM_TEST_SCHEMA_PATH "schema.json"
#endif

const char* const kSampleBlob = R"(
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
)";

// Cùng nội dung nhưng đổi tên nhãn và thứ tự dòng - chứng minh schema-driven.
const char* const kAlternativeBlob = R"(
Chu kỳ: tháng
Mã gói cước: GOI_XYZ99
SMS: 50
Phí đăng ký: 99.000
Cước thuê bao: 55.000
Thoại ngoại mạng: 30 phút
Data tốc độ cao: 5GB
)";

// Ảnh mờ, OCR mất chữ và sai ký tự: chương trình phải trả null chứ không crash.
const char* const kNoisyBlob = R"(
CHUONG TRINH KHUYEN MAI
Ten ma gla 0CS
??? ---
Cuoc dang ky
)";

const std::vector<std::string> kExpectedFields = {
    "packageCode", "registerFee", "monthlyFee", "onnetMinutes", "offnetMinutes", "sms",
    "dataGB",      "youtubeGB",   "spotifyGB",  "cycle",        "allowedPackages"};

/// Provider giả lập: trả token dựng từ một blob text cố định.
class FakeOCRProvider : public ctkm::ocr::OCRProvider {
public:
    explicit FakeOCRProvider(std::string text) : text_(std::move(text)) {}

    std::string name() const override { return "fake"; }

    OCRResult extract(const std::string& imagePath) override {
        calls.push_back(imagePath);
        OCRResult result = OCRResult::fromText(text_, "fake");
        // Không đặt imagePath: fixture không có ảnh thật, tránh morphology cố đọc file.
        return result;
    }

    std::vector<std::string> calls;

private:
    std::string text_;
};

/// Provider luôn lỗi - dùng để kiểm tra pipeline không crash.
class BrokenOCRProvider : public ctkm::ocr::OCRProvider {
public:
    std::string name() const override { return "broken"; }
    OCRResult extract(const std::string& imagePath) override {
        (void)imagePath;
        throw ctkm::ocr::OCRError("model weights hỏng");
    }
};

CTKMExtractor makeExtractor(const std::string& blob = std::string()) {
    CTKMExtractor::Options options;
    options.useMorphology = false;
    options.usePpStructure = false;
    CTKMExtractor extractor(std::string(CTKM_TEST_SCHEMA_PATH), options);
    if (!blob.empty()) {
        extractor.setProvider(std::make_shared<FakeOCRProvider>(blob));
    }
    return extractor;
}

std::vector<std::string> fieldNames(const ExtractionResult& result) {
    std::vector<std::string> names;
    for (const auto& field : result.fields) {
        names.push_back(field.name);
    }
    return names;
}

double numberOf(const ExtractionResult& result, const std::string& name) {
    const auto* field = result.field(name);
    REQUIRE(field != nullptr);
    REQUIRE(field->value.has_value());
    if (std::holds_alternative<long long>(*field->value)) {
        return static_cast<double>(std::get<long long>(*field->value));
    }
    REQUIRE(std::holds_alternative<double>(*field->value));
    return std::get<double>(*field->value);
}

std::string stringOf(const ExtractionResult& result, const std::string& name) {
    const auto* field = result.field(name);
    REQUIRE(field != nullptr);
    REQUIRE(field->value.has_value());
    REQUIRE(std::holds_alternative<std::string>(*field->value));
    return std::get<std::string>(*field->value);
}

std::vector<std::string> listOf(const ExtractionResult& result, const std::string& name) {
    const auto* field = result.field(name);
    REQUIRE(field != nullptr);
    REQUIRE(field->value.has_value());
    REQUIRE(std::holds_alternative<std::vector<std::string>>(*field->value));
    return std::get<std::vector<std::string>>(*field->value);
}

bool isNull(const ExtractionResult& result, const std::string& name) {
    const auto* field = result.field(name);
    return field != nullptr && !field->value.has_value();
}

}  // namespace

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------
TEST_CASE("schema khai báo đủ 11 field theo đúng thứ tự", "[schema]") {
    const auto schema = ctkm::extraction::loadSchema(CTKM_TEST_SCHEMA_PATH);
    REQUIRE(schema.fieldNames() == kExpectedFields);
}

TEST_CASE("mọi field đều có parser và aliases", "[schema]") {
    const auto schema = ctkm::extraction::loadSchema(CTKM_TEST_SCHEMA_PATH);
    for (const auto& spec : schema.fields) {
        REQUIRE_FALSE(spec.parser.empty());
        REQUIRE_FALSE(spec.aliases.empty());
    }
}

TEST_CASE("schema onnetMinutes phải là string + text_parser", "[schema]") {
    // LƯU Ý: ép về number sẽ luôn null hoặc lấy nhầm giá trị của offnetMinutes.
    const auto schema = ctkm::extraction::loadSchema(CTKM_TEST_SCHEMA_PATH);
    bool found = false;
    for (const auto& spec : schema.fields) {
        if (spec.name == "onnetMinutes") {
            found = true;
            REQUIRE(spec.type == "string");
            REQUIRE(spec.parser == "text_parser");
            // Alias "TK thoại" là header thật trên ảnh mẫu.
            REQUIRE(std::find(spec.aliases.begin(), spec.aliases.end(), "TK thoại") !=
                    spec.aliases.end());
        }
    }
    REQUIRE(found);
}

TEST_CASE("schema không tồn tại ném SchemaError", "[schema]") {
    REQUIRE_THROWS_AS(ctkm::extraction::loadSchema("/khong/ton/tai/schema.json"),
                      ctkm::extraction::SchemaError);
}

// ---------------------------------------------------------------------------
// End-to-end với OCR mock
// ---------------------------------------------------------------------------
TEST_CASE("output có đúng các field của schema theo thứ tự", "[e2e]") {
    auto extractor = makeExtractor(kSampleBlob);
    const auto result = extractor.extractFromImage("sample.jpg");
    REQUIRE(fieldNames(result) == kExpectedFields);
}

TEST_CASE("trích xuất đúng giá trị từ blob mẫu", "[e2e]") {
    auto extractor = makeExtractor(kSampleBlob);
    const auto result = extractor.extractFromImage("sample.jpg");

    REQUIRE(stringOf(result, "packageCode") == "CTKMN180X");
    REQUIRE(numberOf(result, "registerFee") == Approx(163636.3636));
    REQUIRE(numberOf(result, "monthlyFee") == Approx(0.0));
    // onnetMinutes là mô tả chính sách, không phải số phút thuần.
    REQUIRE(stringOf(result, "onnetMinutes") == "MP 20p đầu tiên/cuộc gọi");
    REQUIRE(numberOf(result, "offnetMinutes") == Approx(150.0));
    REQUIRE(numberOf(result, "sms") == Approx(100.0));
    REQUIRE(numberOf(result, "dataGB") == Approx(60.0));
    REQUIRE(numberOf(result, "youtubeGB") == Approx(25.0));
    REQUIRE(numberOf(result, "spotifyGB") == Approx(25.0));
    REQUIRE(stringOf(result, "cycle") == "tháng");
    REQUIRE(listOf(result, "allowedPackages") ==
            std::vector<std::string>{"Basic+", "Family", "Corporate++"});
}

TEST_CASE("JSON output không escape tiếng Việt", "[e2e]") {
    auto extractor = makeExtractor(kSampleBlob);
    const auto result = extractor.extractFromImage("sample.jpg");
    const std::string json = result.toJsonString();
    REQUIRE(json.find("tháng") != std::string::npos);
    REQUIRE(json.find("CTKMN180X") != std::string::npos);
}

TEST_CASE("debug report chứa raw text và bảng", "[e2e]") {
    auto extractor = makeExtractor(kSampleBlob);
    const std::string report = extractor.extractFromImage("sample.jpg").debugReport();
    REQUIRE(report.find("OCR raw text") != std::string::npos);
    REQUIRE(report.find("Bảng đã dựng") != std::string::npos);
    REQUIRE(report.find("CTKMN180X") != std::string::npos);
}

TEST_CASE("mẫu CTKM khác dùng alias khác vẫn chạy", "[e2e]") {
    auto extractor = makeExtractor(kAlternativeBlob);
    const auto result = extractor.extractFromImage("other.jpg");

    REQUIRE(stringOf(result, "packageCode") == "GOI_XYZ99");
    REQUIRE(numberOf(result, "registerFee") == Approx(99000.0));
    REQUIRE(numberOf(result, "monthlyFee") == Approx(55000.0));
    REQUIRE(numberOf(result, "offnetMinutes") == Approx(30.0));
    REQUIRE(numberOf(result, "sms") == Approx(50.0));
    REQUIRE(numberOf(result, "dataGB") == Approx(5.0));
    REQUIRE(stringOf(result, "cycle") == "tháng");
    // Không có trong mẫu này -> null chứ không phải lỗi.
    REQUIRE(isNull(result, "youtubeGB"));
    REQUIRE(isNull(result, "allowedPackages"));
}

// ---------------------------------------------------------------------------
// Regression test với header NGUYÊN VĂN của ảnh mẫu thật (BM.12)
// ---------------------------------------------------------------------------
TEST_CASE("bảng BM.12 với header nguyên văn từ ảnh mẫu", "[e2e][regression]") {
    // KHÔNG dùng nhãn đã "làm đẹp" khớp sẵn alias - đây chính là cách duy nhất
    // từng phát hiện ra 4 bug thực tế ở bản Python (monthlyFee/allowedPackages
    // null, onnetMinutes lấy nhầm cột offnetMinutes, dataGB đọc nhầm "(TK533)"
    // trong header thay vì giá trị thật bên dưới).
    const std::vector<std::string> header = {"Tên mã giá OCS",
                                             "Phí đăng ký (VNĐ/tháng)",
                                             "Cước TB (VND tháng)",
                                             "TK thoại",
                                             "Thoại ngoại mạng",
                                             "SMS Trong nước",
                                             "Lưu lượng Data đa hướng (TK533)",
                                             "Ưu đãi data đơn"};
    const std::vector<std::string> data = {"CTKMN180X",
                                           "163,636.3636",
                                           "150,534.213",
                                           "MP 20p đầu tiên",
                                           "150",
                                           "100",
                                           "60GB/tháng, tối đa 8gb/1 ngày",
                                           "Youtube: 25gb/tháng\nSpotify: 25GB/ tháng"};
    const auto table = ctkm::table::tableFromRows({header, data});
    const std::string rawText =
        "Chu kỳ gia hạn: tháng.\n"
        "Gói cước chính được đăng kí: Basic+, Family, Corporate++";

    auto extractor = makeExtractor();
    const auto result = extractor.extractFromTable(table, rawText);

    REQUIRE(stringOf(result, "packageCode") == "CTKMN180X");
    REQUIRE(numberOf(result, "registerFee") == Approx(163636.3636));
    REQUIRE(numberOf(result, "monthlyFee") == Approx(150534.213));
    REQUIRE(stringOf(result, "onnetMinutes") == "MP 20p đầu tiên");
    REQUIRE(numberOf(result, "offnetMinutes") == Approx(150.0));
    REQUIRE(numberOf(result, "sms") == Approx(100.0));
    // Giá trị đúng nằm ở ô DƯỚI header, không phải "533" trong "(TK533)".
    REQUIRE(numberOf(result, "dataGB") == Approx(60.0));
    REQUIRE(numberOf(result, "youtubeGB") == Approx(25.0));
    REQUIRE(numberOf(result, "spotifyGB") == Approx(25.0));
    REQUIRE(stringOf(result, "cycle") == "tháng");
    REQUIRE(listOf(result, "allowedPackages") ==
            std::vector<std::string>{"Basic+", "Family", "Corporate++"});
}

TEST_CASE("collision guard: 2 field cùng một ô thì field score thấp hơn ra null",
          "[e2e][collision]") {
    // Alias "Thoại nội mạng" (onnetMinutes) và header thật "Thoại ngoại mạng"
    // (offnetMinutes) chỉ khác 1-2 ký tự nên fuzzy-match từng khớp nhầm cả hai
    // field vào CÙNG một cột, cho ra cùng giá trị 150.
    const std::vector<std::string> header = {"Tên mã giá OCS", "Thoại ngoại mạng"};
    const std::vector<std::string> data = {"CTKMN180X", "150"};
    const auto table = ctkm::table::tableFromRows({header, data});

    auto extractor = makeExtractor();
    const auto result = extractor.extractFromTable(table);

    // offnetMinutes khớp chính xác (score 1.0) nên thắng và giữ giá trị.
    REQUIRE(numberOf(result, "offnetMinutes") == Approx(150.0));
    // onnetMinutes bị guard đưa về null kèm cảnh báo.
    REQUIRE(isNull(result, "onnetMinutes"));

    bool warned = false;
    for (const auto& warning : result.warnings) {
        if (warning.find("onnetMinutes") != std::string::npos &&
            warning.find("offnetMinutes") != std::string::npos) {
            warned = true;
        }
    }
    REQUIRE(warned);
}

TEST_CASE("field có keyword được loại khỏi collision guard", "[e2e][collision]") {
    // youtubeGB và spotifyGB cùng đọc từ 1 ô gộp nhưng mỗi field tự khoanh vùng
    // theo keyword riêng - đây là thiết kế có chủ đích, không phải xung đột.
    const std::vector<std::string> header = {"Ưu đãi data đơn"};
    const std::vector<std::string> data = {"Youtube: 25gb/tháng\nSpotify: 10GB/ tháng"};
    const auto table = ctkm::table::tableFromRows({header, data});

    auto extractor = makeExtractor();
    const auto result = extractor.extractFromTable(table);

    REQUIRE(numberOf(result, "youtubeGB") == Approx(25.0));
    REQUIRE(numberOf(result, "spotifyGB") == Approx(10.0));
}

// ---------------------------------------------------------------------------
// Xử lý dữ liệu không chuẩn
// ---------------------------------------------------------------------------
TEST_CASE("dữ liệu nhiễu trả null mà không crash", "[robust]") {
    auto extractor = makeExtractor(kNoisyBlob);
    const auto result = extractor.extractFromImage("noisy.jpg");

    REQUIRE(fieldNames(result) == kExpectedFields);
    REQUIRE(isNull(result, "sms"));
    REQUIRE(isNull(result, "allowedPackages"));
    REQUIRE_FALSE(result.warnings.empty());
}

TEST_CASE("OCR không trả token nào thì mọi field là null", "[robust]") {
    auto extractor = makeExtractor("");
    extractor.setProvider(std::make_shared<FakeOCRProvider>(""));
    const auto result = extractor.extractFromImage("blank.jpg");

    for (const auto& name : kExpectedFields) {
        REQUIRE(isNull(result, name));
    }
    REQUIRE(result.tableStrategy == "raw_text");
}

TEST_CASE("OCR ném exception vẫn trả JSON đủ field", "[robust]") {
    CTKMExtractor::Options options;
    options.useMorphology = false;
    options.usePpStructure = false;
    CTKMExtractor extractor(std::string(CTKM_TEST_SCHEMA_PATH), options);
    extractor.setProvider(std::make_shared<BrokenOCRProvider>());

    const auto result = extractor.extractFromImage("broken.jpg");
    for (const auto& name : kExpectedFields) {
        REQUIRE(isNull(result, name));
    }
    bool warned = false;
    for (const auto& warning : result.warnings) {
        if (warning.find("OCR thất bại") != std::string::npos) {
            warned = true;
        }
    }
    REQUIRE(warned);
}

TEST_CASE("field thiếu được đánh dấu missing", "[robust]") {
    auto extractor = makeExtractor("Chu kỳ: tháng");
    const auto result = extractor.extractFromImage("partial.jpg");

    REQUIRE(stringOf(result, "cycle") == "tháng");
    REQUIRE(isNull(result, "sms"));
    REQUIRE(result.field("sms")->source == std::string("missing"));
}

TEST_CASE("nhiễu ký tự OCR trong cụm số vẫn parse được", "[robust]") {
    // "l" thay cho "1" và "O" thay cho "0" ngay trong cụm số.
    const std::string blob = "Cước đăng ký (chưa VAT) l63,636.3636\nƯu đãi SMS 1OO";
    auto extractor = makeExtractor(blob);
    const auto result = extractor.extractFromImage("ocr_noise.jpg");

    REQUIRE(numberOf(result, "registerFee") == Approx(163636.3636));
    REQUIRE(numberOf(result, "sms") == Approx(100.0));
}

// ---------------------------------------------------------------------------
// Bảng dọc / bảng ngang / đổi thứ tự cột
// ---------------------------------------------------------------------------
TEST_CASE("bảng nhãn - giá trị theo hàng", "[table]") {
    const auto table = ctkm::table::tableFromRows({{"Nội dung", "Chi tiết"},
                                                   {"Tên mã giá OCS", "CTKMN180X"},
                                                   {"Cước đăng ký", "163,636.3636"},
                                                   {"Cước thuê bao tháng", "0"},
                                                   {"Ưu đãi SMS", "100"},
                                                   {"Chu kỳ gia hạn", "tháng"},
                                                   {"Gói cước áp dụng", "Basic+, Family"}});
    auto extractor = makeExtractor();
    const auto result = extractor.extractFromTable(table);

    REQUIRE(stringOf(result, "packageCode") == "CTKMN180X");
    REQUIRE(numberOf(result, "registerFee") == Approx(163636.3636));
    REQUIRE(numberOf(result, "monthlyFee") == Approx(0.0));
    REQUIRE(numberOf(result, "sms") == Approx(100.0));
    REQUIRE(stringOf(result, "cycle") == "tháng");
    REQUIRE(listOf(result, "allowedPackages") == std::vector<std::string>{"Basic+", "Family"});
}

TEST_CASE("bảng header trên cùng", "[table]") {
    const auto table = ctkm::table::tableFromRows(
        {{"Mã gói", "Cước đăng ký", "SMS", "Data"}, {"CTKMN180X", "163.636", "100", "60GB"}});
    auto extractor = makeExtractor();
    const auto result = extractor.extractFromTable(table);

    REQUIRE(stringOf(result, "packageCode") == "CTKMN180X");
    REQUIRE(numberOf(result, "registerFee") == Approx(163636.0));
    REQUIRE(numberOf(result, "sms") == Approx(100.0));
    REQUIRE(numberOf(result, "dataGB") == Approx(60.0));
}

TEST_CASE("thứ tự cột không quan trọng", "[table]") {
    const auto table = ctkm::table::tableFromRows(
        {{"SMS", "Mã gói", "Cước đăng ký"}, {"100", "CTKMN180X", "163.636"}});
    auto extractor = makeExtractor();
    const auto result = extractor.extractFromTable(table);

    REQUIRE(stringOf(result, "packageCode") == "CTKMN180X");
    REQUIRE(numberOf(result, "sms") == Approx(100.0));
    REQUIRE(numberOf(result, "registerFee") == Approx(163636.0));
}

TEST_CASE("dựng bảng từ toạ độ token bằng cluster", "[table]") {
    const std::vector<std::pair<std::string, std::string>> rows = {
        {"Tên mã giá OCS", "CTKMN180X"},
        {"Cước đăng ký", "163,636.3636"},
        {"Ưu đãi SMS", "100"}};
    OCRResult ocrResult;
    ocrResult.provider = "fake";
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const double top = 40.0 * static_cast<double>(index);
        ocrResult.tokens.emplace_back(rows[index].first, BoundingBox(10, top, 200, top + 25));
        ocrResult.tokens.emplace_back(rows[index].second, BoundingBox(320, top, 520, top + 25));
    }

    auto extractor = makeExtractor();
    const auto result = extractor.extractFromOcr(ocrResult);

    REQUIRE(result.table.has_value());
    REQUIRE(result.table->columnCount() == 2);
    REQUIRE(result.tableStrategy == "cluster");
    REQUIRE(stringOf(result, "packageCode") == "CTKMN180X");
    REQUIRE(numberOf(result, "registerFee") == Approx(163636.3636));
    REQUIRE(numberOf(result, "sms") == Approx(100.0));
}

TEST_CASE("similarity khớp thuật toán difflib", "[similarity]") {
    // Các giá trị dưới đây lấy từ difflib.SequenceMatcher của Python.
    REQUIRE(ctkm::extraction::similarity("abc", "abc") == Approx(1.0));
    REQUIRE(ctkm::extraction::similarity("thoai noi mang", "thoai ngoai mang") ==
            Approx(0.9333333333).margin(1e-6));
    REQUIRE(ctkm::extraction::similarity("abcd", "wxyz") == Approx(0.0));
}

// ---------------------------------------------------------------------------
// Cắt giá trị tại nhãn của field kế tiếp
// ---------------------------------------------------------------------------
namespace {

/// Dòng raw text lấy NGUYÊN VĂN từ log chạy Paddle+VietOCR trên biểu mẫu BM.12
/// thật: OCR gộp cả HÀNG HEADER của bảng vào MỘT dòng. Ô giá trị của "Cước TB"
/// bị detector bỏ sót (chữ mờ nằm dưới watermark), nên nếu không cắt phần dư
/// thì monthlyFee vớ phải "533" trong "(TK533)" - mã tài khoản của cột Data.
const char* const kFlattenedHeaderRow =
    "hưởng được Tên mã giá ocs (VNĐ/tháng) Phí đăng ký (VND tháng) Cước TB "
    "Thoại ngoại mạng Trong nước SMS Lưu lượng Data đa hướng (TK533) Ưu đãi data dơn";

}  // namespace

TEST_CASE("cutAtNextLabel cắt tại nhãn của field khác", "[cut]") {
    using ctkm::extraction::cutAtNextLabel;

    REQUIRE(cutAtNextLabel("Thoại ngoại mạng Trong nước SMS Lưu lượng Data đa hướng (TK533)",
                           {"Cước TB"}, {"Thoại ngoại mạng", "SMS", "Lưu lượng data"})
                .empty());
    REQUIRE(cutAtNextLabel("180.000 đ", {"Cước TB"}, {"Thoại ngoại mạng"}) == "180.000 đ");
}

TEST_CASE("cutAtNextLabel không cắt tại alias của chính field đó", "[cut]") {
    using ctkm::extraction::cutAtNextLabel;

    REQUIRE(cutAtNextLabel("180.000 (Cước thuê bao tháng)", {"Cước thuê bao tháng"},
                           {"Cước thuê bao tháng", "Thoại ngoại mạng"}) ==
            "180.000 (Cước thuê bao tháng)");
}

TEST_CASE("cutAtNextLabel bỏ qua alias quá ngắn và chỉ cắt ở ranh giới từ", "[cut]") {
    using ctkm::extraction::cutAtNextLabel;

    // "SMS" (3 ký tự) không được cắt cụt một giá trị hợp lệ.
    REQUIRE(cutAtNextLabel("100 SMS/tháng", {"Ưu đãi"}, {"SMS"}) == "100 SMS/tháng");
    // "Youtube" nằm lọt trong "MyYoutubeX" thì không phải một nhãn riêng.
    REQUIRE(cutAtNextLabel("25gb MyYoutubeX", {"Ưu đãi"}, {"Youtube"}) == "25gb MyYoutubeX");
    REQUIRE(cutAtNextLabel("25gb Youtube: 30", {"Ưu đãi"}, {"Youtube"}) == "25gb");
}

TEST_CASE("hàng bảng bị OCR gộp thành một dòng không cho ra số của cột khác", "[cut]") {
    const std::vector<std::string> header = {"Tên mã giá ocs", "Phí đăng ký (VNĐ/tháng)",
                                             "Cước TB (VND tháng)", ""};
    const std::vector<std::string> data = {"CTKMN180X", "163,636.3636", "", "MP 20p đầu tiên"};
    const auto table = ctkm::table::tableFromRows({header, data});

    auto extractor = makeExtractor();
    const auto result = extractor.extractFromTable(table, kFlattenedHeaderRow);

    // Ô giá trị vắng mặt -> null, KHÔNG được bịa ra 533 từ "(TK533)".
    REQUIRE(isNull(result, "monthlyFee"));
}

TEST_CASE("không nhảy xuống dòng dưới khi phần dư bị cắt", "[cut]") {
    const std::string raw =
        std::string(kFlattenedHeaderRow) + "\nMP 20p đầu 60GB/tháng, tôi đa Youtube: 25gb/tháng";

    auto extractor = makeExtractor();
    const auto result = extractor.extractFromText(raw);

    // Nhãn "Cước TB" không chiếm trọn dòng, phần dư chỉ trống vì bị cắt tại
    // nhãn kế tiếp -> lấy dòng dưới là lấy nhầm nội dung cột khác (ra 20).
    REQUIRE(isNull(result, "monthlyFee"));
}

TEST_CASE("nhãn chiếm trọn dòng thì vẫn lấy giá trị ở dòng kế tiếp", "[cut]") {
    auto extractor = makeExtractor();
    const auto result = extractor.extractFromText("Cước thuê bao tháng\n180.000\n");

    REQUIRE(numberOf(result, "monthlyFee") == Approx(180000.0));
}

// ---------------------------------------------------------------------------
// Gộp kết quả nhiều trang (port của TestMergePageResults bên bản Python)
// ---------------------------------------------------------------------------
namespace {

/// Trang giả: field nào có tên trong ``values`` thì coi như trích được.
ExtractionResult makePage(const std::vector<std::pair<std::string, std::string>>& values) {
    ExtractionResult result;
    for (const char* name : {"packageCode", "registerFee", "monthlyFee", "onnetMinutes",
                             "offnetMinutes", "sms", "dataGB", "youtubeGB", "spotifyGB",
                             "cycle", "allowedPackages"}) {
        ctkm::extraction::FieldResult field;
        field.name = name;
        for (const auto& entry : values) {
            if (entry.first == name) {
                field.value = FieldValue(entry.second);
                field.score = 0.9;
                field.source = ctkm::extraction::kSourceTable;
            }
        }
        result.fields.push_back(field);
    }
    return result;
}

}  // namespace

TEST_CASE("chọn trang nhiều field nhất làm trang chính", "[merge]") {
    using ctkm::extraction::mergePageResults;
    const auto trong = makePage({});
    const auto coBang = makePage({{"packageCode", "CTKMN180X"}, {"offnetMinutes", "150"},
                                  {"sms", "100"}});

    const auto merged = mergePageResults({{1, trong}, {2, coBang}, {3, trong}});

    REQUIRE(merged.page.has_value());
    REQUIRE(*merged.page == 2);
    REQUIRE(merged.pagesProcessed == 3);
    REQUIRE(stringOf(merged, "packageCode") == "CTKMN180X");
    REQUIRE(*merged.field("packageCode")->sourcePage == 2);
}

TEST_CASE("bù field thiếu từ trang khác", "[merge]") {
    using ctkm::extraction::mergePageResults;
    const auto trang1 = makePage({{"packageCode", "CTKMN180X"}, {"offnetMinutes", "150"}});
    const auto trang2 = makePage({{"cycle", "tháng"}});

    const auto merged = mergePageResults({{1, trang1}, {2, trang2}});

    REQUIRE(*merged.page == 1);
    REQUIRE(stringOf(merged, "cycle") == "tháng");
    REQUIRE(*merged.field("cycle")->sourcePage == 2);

    bool warned = false;
    for (const auto& warning : merged.warnings) {
        if (warning.find("trang 2") != std::string::npos) {
            warned = true;
        }
    }
    REQUIRE(warned);
}

TEST_CASE("không ghi đè giá trị của trang chính", "[merge]") {
    using ctkm::extraction::mergePageResults;
    const auto chinh = makePage({{"packageCode", "CTKMN180X"}, {"offnetMinutes", "150"},
                                 {"sms", "100"}});
    const auto phu = makePage({{"packageCode", "SAI_BEN_TRANG_KHAC"}});

    const auto merged = mergePageResults({{1, chinh}, {2, phu}});

    REQUIRE(stringOf(merged, "packageCode") == "CTKMN180X");
    REQUIRE(*merged.field("packageCode")->sourcePage == 1);
}

TEST_CASE("mọi trang đều rỗng vẫn trả đủ field null", "[merge]") {
    using ctkm::extraction::mergePageResults;
    const auto merged = mergePageResults({{1, makePage({})}, {2, makePage({})}});

    REQUIRE(merged.pagesProcessed == 2);
    REQUIRE(merged.fields.size() == 11);
    for (const auto& field : merged.fields) {
        REQUIRE_FALSE(field.found());
    }
}

TEST_CASE("không có trang nào thì không crash", "[merge]") {
    const auto merged = ctkm::extraction::mergePageResults({});
    REQUIRE(merged.pagesProcessed == 0);
    REQUIRE(merged.fields.empty());
}

TEST_CASE("hoà điểm thì ưu tiên trang số nhỏ", "[merge]") {
    using ctkm::extraction::mergePageResults;
    const auto a = makePage({{"packageCode", "A"}});
    const auto b = makePage({{"packageCode", "B"}});

    REQUIRE(*mergePageResults({{5, b}, {2, a}}).page == 2);
    REQUIRE(stringOf(mergePageResults({{5, b}, {2, a}}), "packageCode") == "A");
}

// ---------------------------------------------------------------------------
// Tự chọn cấu hình tiền xử lý
// ---------------------------------------------------------------------------
namespace {

/// Provider giả trả text KHÁC NHAU tuỳ theo có nhị phân hoá hay không.
class PreprocessAwareProvider : public ctkm::ocr::OCRProvider {
public:
    PreprocessAwareProvider(std::string whenBinarized, std::string whenNot)
        : whenBinarized_(std::move(whenBinarized)), whenNot_(std::move(whenNot)) {}

    std::string name() const override { return "fake_preprocess"; }

    bool setPreprocess(const ctkm::ocr::PreprocessConfig& config) override {
        binarize_ = config.adaptiveThreshold;
        return true;
    }

    ctkm::ocr::OCRResult extract(const std::string& imagePath) override {
        seen.push_back(binarize_);
        ctkm::ocr::OCRResult result;
        result.imagePath = imagePath;
        result.provider = name();
        std::istringstream stream(binarize_ ? whenBinarized_ : whenNot_);
        std::string line;
        int index = 0;
        while (std::getline(stream, line)) {
            if (!line.empty()) {
                result.tokens.emplace_back(line, BoundingBox(0, 30.0 * index, 400,
                                                             30.0 * index + 25));
                ++index;
            }
        }
        return result;
    }

    std::vector<bool> seen;

private:
    std::string whenBinarized_, whenNot_;
    bool binarize_ = true;
};

const char* const kÍt = "Tên mã giá OCS: GOI_A";
const char* const kNhiều = "Tên mã giá OCS: GOI_B\nƯu đãi SMS: 100\nChu kỳ: tháng";

}  // namespace

TEST_CASE("mặc định chạy cả hai cấu hình rồi giữ kết quả tốt hơn", "[binarize]") {
    auto provider = std::make_shared<PreprocessAwareProvider>(kÍt, kNhiều);
    CTKMExtractor::Options options;
    options.usePpStructure = false;
    CTKMExtractor extractor(std::string(CTKM_TEST_SCHEMA_PATH), options);
    extractor.setProvider(provider);

    const auto result = extractor.extractFromImage("a.png");

    REQUIRE(provider->seen == std::vector<bool>{true, false});
    REQUIRE(stringOf(result, "packageCode") == "GOI_B");
    REQUIRE(numberOf(result, "sms") == Approx(100.0));
}

TEST_CASE("giữ kết quả tốt hơn kể cả khi nó ở lượt đầu", "[binarize]") {
    auto provider = std::make_shared<PreprocessAwareProvider>(kNhiều, kÍt);
    CTKMExtractor::Options options;
    options.usePpStructure = false;
    CTKMExtractor extractor(std::string(CTKM_TEST_SCHEMA_PATH), options);
    extractor.setProvider(provider);

    REQUIRE(stringOf(extractor.extractFromImage("a.png"), "packageCode") == "GOI_B");
}

TEST_CASE("ép nhị phân hoá thì chỉ chạy một lượt", "[binarize]") {
    for (const bool forced : {true, false}) {
        auto provider = std::make_shared<PreprocessAwareProvider>(kÍt, kNhiều);
        CTKMExtractor::Options options;
        options.usePpStructure = false;
        options.binarize = forced;
        CTKMExtractor extractor(std::string(CTKM_TEST_SCHEMA_PATH), options);
        extractor.setProvider(provider);

        extractor.extractFromImage("a.png");

        REQUIRE(provider->seen == std::vector<bool>{forced});
    }
}
