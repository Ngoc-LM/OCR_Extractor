// Port của ``ctkm_extractor/tests/test_morphology.py``.
//
// Dùng ảnh bảng viền TỔNG HỢP (vẽ bằng OpenCV) thay vì file ảnh thật, để test
// độc lập, không cần file ngoài mà vẫn chạy qua đúng pipeline ảnh thật.

#include <catch2/catch_test_macros.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include "ocr/OCRProvider.hpp"
#include "table/Morphology.hpp"
#include "table/TableBuilder.hpp"

using ctkm::ocr::BoundingBox;
using ctkm::ocr::OCRToken;
using ctkm::table::MorphologyTableRecognizer;
using ctkm::table::TableMorphologyUnavailableError;

namespace {

/// Vẽ 1 ảnh trắng có lưới bảng viền đen tại các toạ độ cho trước.
cv::Mat drawBorderedTable(const std::vector<int>& rowBoundaries,
                          const std::vector<int>& colBoundaries, int height = 400,
                          int width = 900) {
    cv::Mat image(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
    for (const int y : rowBoundaries) {
        cv::line(image, cv::Point(0, y), cv::Point(width, y), cv::Scalar(0, 0, 0), 2);
    }
    for (const int x : colBoundaries) {
        cv::line(image, cv::Point(x, 0), cv::Point(x, height), cv::Scalar(0, 0, 0), 2);
    }
    return image;
}

OCRToken token(const std::string& text, double x1, double x2, double y1, double y2) {
    return OCRToken(text, BoundingBox(x1, y1, x2, y2));
}

/// Tra text của ô theo toạ độ (hàng, cột).
std::string cellText(const ctkm::table::Table& table, int row, int col) {
    for (const auto& cell : table.cells()) {
        if (cell.row == row && cell.col == col) {
            return cell.text;
        }
    }
    return std::string();
}

}  // namespace

// ---------------------------------------------------------------------------
// detectGrid
// ---------------------------------------------------------------------------
TEST_CASE("detect_grid ném lỗi khi không đọc được file", "[morphology][grid]") {
    REQUIRE_THROWS_AS(ctkm::table::detectGrid(std::string("/khong/ton/tai/anh.jpg")),
                      TableMorphologyUnavailableError);
}

TEST_CASE("detect_grid ném lỗi với ảnh rỗng", "[morphology][grid]") {
    REQUIRE_THROWS_AS(ctkm::table::detectGrid(cv::Mat()), TableMorphologyUnavailableError);
}

TEST_CASE("detect_grid đếm đúng số đường ngang/dọc", "[morphology][grid]") {
    // Bảng 2 hàng x 3 cột: 3 đường ngang, 4 đường dọc.
    const std::vector<int> rows = {40, 200, 360};
    const std::vector<int> cols = {60, 300, 550, 840};
    const cv::Mat image = drawBorderedTable(rows, cols);

    const ctkm::table::Grid grid = ctkm::table::detectGrid(image);
    REQUIRE(grid.rowLines.size() == rows.size());
    REQUIRE(grid.colLines.size() == cols.size());
}

// ---------------------------------------------------------------------------
// buildTable phải dò đường kẻ trên ĐÚNG ảnh đã sinh ra token
// ---------------------------------------------------------------------------
TEST_CASE("buildTable dùng ảnh đã tiền xử lý thay vì đọc lại ảnh gốc",
          "[morphology][regression]") {
    // Regression: tiền xử lý xoay/phóng to ảnh nên toạ độ token khác ảnh gốc.
    // Trước khi sửa, buildTable đọc lại ảnh gốc từ imagePath: với ảnh scan bị
    // nghiêng thì không dò được đường kẻ ngang nào (hàng=0) nên morphology - mức
    // dựng bảng tốt nhất - bị vô hiệu ngay trên loại ảnh cần nó nhất.
    const cv::Mat processed = drawBorderedTable({40, 140, 240}, {30, 400, 900});

    // Ảnh "gốc" trên đĩa trắng trơn - không có đường kẻ nào để dò.
    const std::string originalPath =
        (std::filesystem::temp_directory_path() / "ctkm_original_blank.png").string();
    cv::imwrite(originalPath, cv::Mat(400, 900, CV_8UC3, cv::Scalar(255, 255, 255)));

    ctkm::ocr::OCRResult ocrResult;
    ocrResult.provider = "fake";
    ocrResult.imagePath = originalPath;
    ocrResult.processedImage = processed;
    ocrResult.tokens = {token("Nội dung", 60, 200, 70, 100),
                        token("Chi tiết", 430, 560, 70, 100),
                        token("Chu kỳ", 60, 200, 170, 200),
                        token("tháng", 430, 520, 170, 200)};

    ctkm::table::BuildOptions options;
    options.usePpStructure = false;
    const auto result = ctkm::table::buildTable(ocrResult, options);

    REQUIRE(result.strategy == std::string(ctkm::table::kStrategyMorphology));
    REQUIRE(result.table.has_value());
    REQUIRE(result.table->columnCount() == 2);

    std::error_code code;
    std::filesystem::remove(originalPath, code);
}

TEST_CASE("buildTable vẫn đọc ảnh gốc khi không có ảnh tiền xử lý",
          "[morphology][regression]") {
    const std::string imagePath =
        (std::filesystem::temp_directory_path() / "ctkm_table_border.png").string();
    cv::imwrite(imagePath, drawBorderedTable({40, 140, 240}, {30, 400, 900}));

    ctkm::ocr::OCRResult ocrResult;
    ocrResult.provider = "fake";
    ocrResult.imagePath = imagePath;
    ocrResult.tokens = {token("Nội dung", 60, 200, 70, 100),
                        token("Chi tiết", 430, 560, 70, 100),
                        token("Chu kỳ", 60, 200, 170, 200),
                        token("tháng", 430, 520, 170, 200)};

    ctkm::table::BuildOptions options;
    options.usePpStructure = false;
    const auto result = ctkm::table::buildTable(ocrResult, options);

    REQUIRE(result.strategy == std::string(ctkm::table::kStrategyMorphology));

    std::error_code code;
    std::filesystem::remove(imagePath, code);
}

// ---------------------------------------------------------------------------
// MorphologyTableRecognizer
// ---------------------------------------------------------------------------
TEST_CASE("ném lỗi khi quá ít đường kẻ", "[morphology]") {
    // Ảnh trắng trơn - không có đường kẻ nào.
    const cv::Mat image(200, 400, CV_8UC3, cv::Scalar(255, 255, 255));
    MorphologyTableRecognizer recognizer;
    REQUIRE_THROWS_AS(recognizer.recognize(image, {}), TableMorphologyUnavailableError);
}

TEST_CASE("nhận đúng hàng header và hàng dữ liệu theo cột", "[morphology]") {
    // Bảng 2 hàng (header + data) x 3 cột, giống cấu trúc bảng CTKM thật.
    const cv::Mat image = drawBorderedTable({40, 140, 240}, {30, 250, 500, 750});
    const std::vector<OCRToken> tokens = {
        token("Mã gói", 60, 200, 70, 100),      token("Phí đăng ký", 280, 460, 70, 100),
        token("Data GB", 530, 700, 70, 100),    token("N180X", 60, 150, 170, 200),
        token("163636", 280, 380, 170, 200),    token("60", 530, 570, 170, 200)};

    MorphologyTableRecognizer recognizer;
    const auto table = recognizer.recognize(image, tokens);

    REQUIRE(table.rowCount() == 2);
    REQUIRE(table.columnCount() == 3);
    REQUIRE(cellText(table, 0, 0) == "Mã gói");
    REQUIRE(cellText(table, 0, 1) == "Phí đăng ký");
    REQUIRE(cellText(table, 0, 2) == "Data GB");
    REQUIRE(cellText(table, 1, 0) == "N180X");
    REQUIRE(cellText(table, 1, 1) == "163636");
    REQUIRE(cellText(table, 1, 2) == "60");
}

TEST_CASE("ném lỗi khi không token nào khớp hàng nào", "[morphology]") {
    const cv::Mat image = drawBorderedTable({40, 140, 240}, {30, 250, 500, 750});
    // Token nằm ngoài mọi hàng đã dò (y quá lớn).
    const std::vector<OCRToken> tokens = {token("lạc", 60, 150, 900, 930)};

    MorphologyTableRecognizer recognizer;
    REQUIRE_THROWS_AS(recognizer.recognize(image, tokens), TableMorphologyUnavailableError);
}

TEST_CASE("token dữ liệu tràn ngoài band header vẫn vào đúng cột giá trị",
          "[morphology][regression]") {
    // LƯU Ý (bug thực tế): band cột chỉ rộng bằng chữ trong header, nên giá trị
    // dài hơn tiêu đề cột có phần không chồng lấn band nào. Nếu chọn cột bằng
    // "overlap lớn nhất" với ngưỡng khởi tạo -1.0, token đó rơi về cột 0 và dính
    // vào ô nhãn ("Ưu đãi Data" nuốt "tối đa 8gb/1 ngày" -> dataGB ra 8 thay vì 60).
    const cv::Mat image = drawBorderedTable({40, 140, 240}, {30, 400, 900});
    const std::vector<OCRToken> tokens = {
        token("Nội dung", 60, 200, 70, 100),
        // Header cột giá trị hẹp: chỉ từ x=430 tới x=560.
        token("Chi tiết", 430, 560, 70, 100),
        token("Ưu đãi Data", 60, 250, 170, 200),
        // Giá trị dài, phần đuôi nằm hoàn toàn bên phải band header.
        token("60GB/tháng,", 430, 600, 170, 200),
        token("tối đa 8gb/1 ngày", 620, 860, 170, 200)};

    MorphologyTableRecognizer recognizer;
    const auto table = recognizer.recognize(image, tokens);

    REQUIRE(cellText(table, 1, 0) == "Ưu đãi Data");
    REQUIRE(cellText(table, 1, 1) == "60GB/tháng, tối đa 8gb/1 ngày");
}

TEST_CASE("hàng cuối thiếu đường viền dưới vẫn được giữ", "[morphology][regression]") {
    const cv::Mat image = drawBorderedTable({40, 140, 240}, {30, 400, 900});
    const std::vector<OCRToken> tokens = {
        token("Nội dung", 60, 200, 70, 100), token("Chi tiết", 430, 560, 70, 100),
        token("Chu kỳ", 60, 200, 170, 200), token("tháng", 430, 520, 170, 200),
        // Hàng cuối: nằm DƯỚI đường kẻ cuối cùng (y=240) nhưng còn trong phạm vi
        // một bước hàng, tức là hàng chưa được đóng viền.
        token("Gói cước áp dụng", 60, 300, 270, 300),
        token("Basic+, Family", 430, 640, 270, 300)};

    MorphologyTableRecognizer recognizer;
    const auto table = recognizer.recognize(image, tokens);

    REQUIRE(table.rowCount() == 3);
    REQUIRE(cellText(table, 2, 0) == "Gói cước áp dụng");
    REQUIRE(cellText(table, 2, 1) == "Basic+, Family");
}

TEST_CASE("chữ ở chân trang không bị kéo vào bảng", "[morphology][regression]") {
    const cv::Mat image = drawBorderedTable({40, 140, 240}, {30, 400, 900});
    const std::vector<OCRToken> tokens = {
        token("Nội dung", 60, 200, 70, 100), token("Chi tiết", 430, 560, 70, 100),
        token("Chu kỳ", 60, 200, 170, 200), token("tháng", 430, 520, 170, 200),
        token("Trang 1/2 - biểu mẫu BM.12", 60, 400, 380, 399)};

    MorphologyTableRecognizer recognizer;
    const auto table = recognizer.recognize(image, tokens);

    REQUIRE(table.rowCount() == 2);
    for (const auto& cell : table.cells()) {
        REQUIRE(cell.text.find("Trang 1/2") == std::string::npos);
    }
}

TEST_CASE("trailing_row_limit bằng một bước hàng sau đường kẻ cuối", "[morphology]") {
    const auto limit = ctkm::table::trailingRowLimit({40.0, 140.0, 240.0});
    REQUIRE(limit.has_value());
    REQUIRE(*limit == 340.0);
    // Ít hơn 2 đường kẻ thì không suy được bước hàng.
    REQUIRE_FALSE(ctkm::table::trailingRowLimit({40.0}).has_value());
}

TEST_CASE("best_band_nearest chọn band gần nhất khi không chồng lấn", "[morphology]") {
    const std::vector<std::pair<double, double>> bands = {{60.0, 190.0}, {520.0, 620.0}};
    // Token nằm hoàn toàn bên phải cả 2 band -> phải chọn band 1 (gần nhất).
    REQUIRE(ctkm::table::bestBandNearest(BoundingBox(650, 0, 700, 20), bands) == 1);
    // Token chồng lấn band 0 -> chọn band 0.
    REQUIRE(ctkm::table::bestBandNearest(BoundingBox(70, 0, 150, 20), bands) == 0);
    // Token bên trái mọi band -> chọn band 0.
    REQUIRE(ctkm::table::bestBandNearest(BoundingBox(0, 0, 20, 20), bands) == 0);
}


// ---------------------------------------------------------------------------
// Bảng lồng bảng
// ---------------------------------------------------------------------------
namespace {

/// Bảng NGOÀI 4 cột phủ cả trang, bên trong chứa bảng con 6 cột.
///
/// Tái hiện cấu trúc biểu mẫu BM.12 thật: bảng CTKM nằm lồng trong một ô của
/// bảng bao ngoài.
cv::Mat drawNestedTable() {
    cv::Mat image(1200, 1600, CV_8UC3, cv::Scalar(255, 255, 255));
    for (const int y : {100, 300, 800, 1100}) {
        cv::line(image, cv::Point(60, y), cv::Point(1540, y), cv::Scalar(0, 0, 0), 3);
    }
    for (const int x : {60, 200, 400, 1540}) {
        cv::line(image, cv::Point(x, 100), cv::Point(x, 1100), cv::Scalar(0, 0, 0), 3);
    }
    for (const int y : {360, 520, 740}) {
        cv::line(image, cv::Point(430, y), cv::Point(1500, y), cv::Scalar(0, 0, 0), 3);
    }
    for (const int x : {430, 610, 790, 970, 1150, 1330, 1500}) {
        cv::line(image, cv::Point(x, 360), cv::Point(x, 740), cv::Scalar(0, 0, 0), 3);
    }
    return image;
}

}  // namespace

TEST_CASE("select_densest_band tìm đúng bảng con", "[morphology][nested]") {
    const ctkm::table::Grid grid = ctkm::table::detectGrid(drawNestedTable());
    const auto band = ctkm::table::selectDensestBand(grid.rowLines, grid.colSegments);

    REQUIRE(band.has_value());
    // Dải được chọn phải nằm gọn trong ô lớn của bảng ngoài (y 300..800).
    REQUIRE(band->first >= 300.0);
    REQUIRE(band->first <= 380.0);
    REQUIRE(band->second >= 700.0);
    REQUIRE(band->second <= 800.0);
}

TEST_CASE("select_densest_band trả nullopt khi chỉ có một bảng", "[morphology][nested]") {
    const ctkm::table::Grid grid = ctkm::table::detectGrid(drawBorderedTable({40, 140, 240},
                                                                            {30, 400, 900}));
    REQUIRE_FALSE(ctkm::table::selectDensestBand(grid.rowLines, grid.colSegments).has_value());
}

TEST_CASE("columns_in_band loại đường kẻ của bảng ngoài", "[morphology][nested]") {
    const ctkm::table::Grid grid = ctkm::table::detectGrid(drawNestedTable());
    const auto band = ctkm::table::selectDensestBand(grid.rowLines, grid.colSegments);
    REQUIRE(band.has_value());

    const auto columns = ctkm::table::columnsInBand(grid.colSegments, band->first, band->second);
    // 7 đường dọc của bảng con; 4 đường của bảng ngoài kéo dài vượt xa nên bị loại.
    REQUIRE(columns.size() == 7);
    REQUIRE(columns.front() > 400.0);
}

TEST_CASE("recognize dùng bảng con chứ không phải bảng ngoài", "[morphology][nested]") {
    const cv::Mat image = drawNestedTable();
    const std::vector<OCRToken> tokens = {
        // Text của bảng NGOÀI - phải bị loại.
        token("Chính sách được hưởng", 210, 395, 400, 440),
        token("TT", 70, 190, 150, 190),
        // Header + dữ liệu của bảng con (6 cột).
        token("Mã gói", 440, 600, 400, 440),
        token("Phí ĐK", 620, 780, 400, 440),
        token("Cước TB", 800, 960, 400, 440),
        token("SMS", 980, 1140, 400, 440),
        token("Data", 1160, 1320, 400, 440),
        token("Chu kỳ", 1340, 1490, 400, 440),
        token("N180X", 440, 600, 600, 640),
        token("163636", 620, 780, 600, 640),
        token("150534", 800, 960, 600, 640),
        token("100", 980, 1140, 600, 640),
        token("60GB", 1160, 1320, 600, 640),
        token("tháng", 1340, 1490, 600, 640)};

    const auto table = MorphologyTableRecognizer().recognize(image, tokens);

    REQUIRE(table.rowCount() == 2);
    REQUIRE(table.columnCount() == 6);  // cột của bảng CON, không phải bảng ngoài
    REQUIRE(cellText(table, 0, 0) == "Mã gói");
    REQUIRE(cellText(table, 1, 0) == "N180X");
    REQUIRE(cellText(table, 1, 3) == "100");
    for (const auto& cell : table.cells()) {
        REQUIRE(cell.text.find("Chính sách") == std::string::npos);
    }
}
