// Port của ``ctkm_extractor/table/reconstruct.py``.
//
// Mô hình dữ liệu bảng (Table/TableCell) + fallback dựng bảng bằng cách cluster
// bounding box: gom token thành hàng theo trục y, rồi thành cột theo trục x.
// Cách này không cần model nhưng dễ sai với ô merge dọc, nên chỉ dùng khi
// morphology (và PP-Structure) thất bại.
#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ocr/OCRProvider.hpp"

namespace ctkm::table {

using ocr::BoundingBox;
using ocr::OCRToken;

/// Token có bề rộng lớn hơn tỉ lệ này so với bảng thì không dùng để dựng biên cột.
constexpr double kWideTokenRatio = 0.45;

/// Ngưỡng tối thiểu phần diện tích token nằm trong ô để coi là thuộc ô đó.
constexpr double kMinCellOverlap = 0.30;

/// Một ô của bảng, có hỗ trợ ô gộp (``rowSpan``/``colSpan``).
struct TableCell {
    int row = 0;
    int col = 0;
    int rowSpan = 1;
    int colSpan = 1;
    std::optional<BoundingBox> box;
    std::string text;
    std::vector<OCRToken> tokens;

    TableCell() = default;
    TableCell(int cellRow, int cellCol, std::string cellText = std::string(),
              std::optional<BoundingBox> cellBox = std::nullopt, int cellRowSpan = 1,
              int cellColSpan = 1);

    bool isEmpty() const;
    /// True nếu ô này phủ toạ độ ``(row, col)``.
    bool covers(int targetRow, int targetCol) const;
};

/// Bảng đã dựng: danh sách ô + kích thước lưới.
class Table {
public:
    Table() = default;
    Table(std::vector<TableCell> cells, int rows, int cols, std::string source = "unknown",
          double confidence = 0.0);

    const std::vector<TableCell>& cells() const { return cells_; }
    std::vector<TableCell>& cells() { return cells_; }
    int rowCount() const { return nRows_; }
    int columnCount() const { return nCols_; }
    const std::string& source() const { return source_; }
    double confidence() const { return confidence_; }

    /// True nếu mọi ô đều rỗng.
    bool isEmpty() const;
    /// Ô phủ toạ độ ``(row, col)``; ô gộp trả về chính nó ở mọi toạ độ nó phủ.
    const TableCell* cellAt(int row, int col) const;
    /// Các ô thuộc một hàng, sắp theo cột.
    std::vector<const TableCell*> rowCells(int row) const;
    /// Các ô thuộc một cột, sắp theo hàng.
    std::vector<const TableCell*> columnCells(int col) const;
    /// Danh sách ô đã sắp theo (hàng, cột).
    std::vector<const TableCell*> sortedCells() const;
    /// Lưới text đầy đủ; ô gộp lặp lại text ở mọi vị trí nó phủ.
    std::vector<std::vector<std::string>> toGrid() const;
    /// Chuỗi dạng bảng ASCII để in khi chạy ``--debug``.
    std::string render(std::size_t maxWidth = 40) const;

private:
    std::vector<TableCell> cells_;
    int nRows_ = 0;
    int nCols_ = 0;
    std::string source_ = "unknown";
    double confidence_ = 0.0;
};

/// Ghép text các token trong một ô theo thứ tự đọc.
std::string joinTokens(const std::vector<OCRToken>& tokens);

/// Gán token OCR vào ô theo diện tích chồng lấn; trả số token gán được.
int assignTokensToCells(std::vector<TableCell>& cells, const std::vector<OCRToken>& tokens,
                        double minOverlap = kMinCellOverlap);

/// Gộp các khoảng [x1, x2] chồng lấn hoặc cách nhau dưới ``minGap``.
std::vector<std::pair<double, double>> mergeIntervals(
    const std::vector<std::pair<double, double>>& intervals, double minGap);

/// Suy ra biên các cột từ hình chiếu bounding box lên trục x.
std::vector<std::pair<double, double>> detectColumnBands(const std::vector<OCRToken>& tokens);

/// Chỉ số cột có phần chồng lấn theo trục x lớn nhất với ``box``.
///
/// LƯU Ý (port 1:1): bản Python của tầng cluster giữ nguyên khởi tạo
/// ``best_overlap = -1.0``. Hệ quả là token nằm cách MỌI band quá 1 pixel sẽ rơi
/// về cột 0. Tầng morphology đã sửa lỗi này bằng nhánh "band gần nhất"
/// (:func:`ctkm::table::bestBandNearest`), còn ở đây giữ nguyên hành vi bản
/// Python để hai bản cho cùng kết quả.
int bestBand(const BoundingBox& box, const std::vector<std::pair<double, double>>& bands);

/// Dựng bảng bằng cách cluster bounding box: hàng theo y, cột theo x.
/// Trả ``std::nullopt`` nếu không đủ dữ liệu để dựng bảng có nghĩa.
std::optional<Table> clusterTokensToTable(const std::vector<OCRToken>& tokens,
                                          double lineOverlapThreshold = 0.4,
                                          int maxColumns = 24);

/// Dựng ``Table`` từ lưới text thuần - tiện cho unit test và fixture.
Table tableFromRows(const std::vector<std::vector<std::string>>& rows,
                    const std::string& source = "manual");

}  // namespace ctkm::table
