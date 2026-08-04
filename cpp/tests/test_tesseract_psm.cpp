// Test riêng cho logic chọn chế độ phân đoạn (psm) của provider fallback.
//
// Không cần binary tesseract: selectBestPsmRun nhận callback chạy một lượt, nên
// test bơm thẳng số token giả vào.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "ocr/OCRProvider.hpp"
#include "ocr/TesseractOCR.hpp"

using ctkm::ocr::BoundingBox;
using ctkm::ocr::OCRToken;
using ctkm::ocr::PsmRun;
using ctkm::ocr::selectBestPsmRun;

namespace {

std::vector<OCRToken> makeTokens(std::size_t count, double confidence) {
    std::vector<OCRToken> tokens;
    for (std::size_t index = 0; index < count; ++index) {
        tokens.emplace_back("t" + std::to_string(index), BoundingBox(0, 0, 10, 10), confidence);
    }
    return tokens;
}

}  // namespace

TEST_CASE("thử hết mọi psm và giữ kết quả tốt nhất", "[tesseract][regression]") {
    // Regression: psm tốt nhất nằm CUỐI danh sách vẫn phải được thử. Bản đầu dừng
    // sớm khi psm đầu tiên đạt >= 8 token và confidence >= 0.6 - trên biểu mẫu
    // BM.12 thật, psm 4 đạt ngưỡng đó ngay nhưng không đọc được bảng CTKM, còn
    // psm 6 (thử sau cùng) mới là psm đọc được.
    std::vector<int> calls;
    const PsmRun best = selectBestPsmRun({4, 11, 6}, [&](int psm) {
        calls.push_back(psm);
        const std::size_t counts = psm == 4 ? 8 : (psm == 11 ? 3 : 20);
        return makeTokens(counts, 0.9);
    });

    REQUIRE(calls == std::vector<int>{4, 11, 6});  // không dừng sớm
    REQUIRE(best.psm == 6);
    REQUIRE(best.tokens.size() == 20);
}

TEST_CASE("ép một psm cụ thể thì chỉ chạy một lượt", "[tesseract]") {
    std::vector<int> calls;
    const PsmRun best = selectBestPsmRun({6}, [&](int psm) {
        calls.push_back(psm);
        return makeTokens(5, 0.8);
    });

    REQUIRE(calls == std::vector<int>{6});
    REQUIRE(best.psm == 6);
}

TEST_CASE("một psm lỗi không làm hỏng cả lượt", "[tesseract]") {
    std::vector<std::string> errors;
    const PsmRun best = selectBestPsmRun(
        {4, 11, 6},
        [](int psm) -> std::vector<OCRToken> {
            if (psm == 4) {
                throw std::runtime_error("engine lỗi");
            }
            return makeTokens(6, 0.7);
        },
        &errors);

    REQUIRE(best.tokens.size() == 6);
    REQUIRE(errors.size() == 1);
    REQUIRE(errors.front().find("psm 4") != std::string::npos);
}

TEST_CASE("không lượt nào chạy được thì psm = -1", "[tesseract]") {
    const PsmRun best = selectBestPsmRun({4, 11, 6}, [](int) -> std::vector<OCRToken> {
        throw std::runtime_error("engine lỗi");
    });
    REQUIRE(best.psm == -1);
    REQUIRE(best.tokens.empty());
}

// ---------------------------------------------------------------------------
// Chọn gói ngôn ngữ
// ---------------------------------------------------------------------------
namespace {

/// Trả về hàm "gói này dùng được không" từ một danh sách gói giả định.
std::function<bool(const std::string&)> installed(std::vector<std::string> packages) {
    return [packages = std::move(packages)](const std::string& language) {
        // Tesseract Init() nhận dạng ghép "vie+eng"; thiếu MỘT phần là hỏng cả cụm.
        std::size_t start = 0;
        while (start <= language.size()) {
            const std::size_t plus = language.find('+', start);
            const std::string part = language.substr(
                start, plus == std::string::npos ? std::string::npos : plus - start);
            if (!part.empty() &&
                std::find(packages.begin(), packages.end(), part) == packages.end()) {
                return false;
            }
            if (plus == std::string::npos) {
                break;
            }
            start = plus + 1;
        }
        return true;
    };
}

}  // namespace

TEST_CASE("pickLanguage ưu tiên gói tiếng Việt", "[language]") {
    using ctkm::ocr::pickLanguage;

    REQUIRE(pickLanguage("", installed({"eng", "vie"})) == "vie");
    // Không có 'vie' đơn lẻ thì rơi xuống 'vie+eng', rồi mới tới 'eng'.
    REQUIRE(pickLanguage("", installed({"eng"})) == "eng");
}

TEST_CASE("pickLanguage tôn trọng ngôn ngữ được yêu cầu", "[language]") {
    using ctkm::ocr::pickLanguage;

    REQUIRE(pickLanguage("deu", installed({"deu", "eng", "vie"})) == "deu");
    // Yêu cầu gói không có -> cảnh báo rồi rơi về mặc định, KHÔNG ném exception.
    REQUIRE(pickLanguage("deu", installed({"eng", "vie"})) == "vie");
}

TEST_CASE("pickLanguage xử lý ngôn ngữ ghép", "[language]") {
    using ctkm::ocr::pickLanguage;

    REQUIRE(pickLanguage("vie+eng", installed({"eng", "vie"})) == "vie+eng");
    // Thiếu một phần của cụm ghép thì cả cụm không dùng được.
    REQUIRE(pickLanguage("vie+deu", installed({"eng", "vie"})) == "vie");
}

TEST_CASE("pickLanguage không treo khi không có gói nào", "[language]") {
    using ctkm::ocr::pickLanguage;

    // Không có gì dùng được: trả 'eng' để tầng trên Init() rồi báo lỗi tử tế.
    REQUIRE(pickLanguage("", installed({})) == "eng");
    REQUIRE(pickLanguage("vie", installed({})) == "eng");
}
