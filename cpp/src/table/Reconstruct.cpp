#include "table/Reconstruct.hpp"

#include <algorithm>
#include <map>

#include "util/Log.hpp"
#include "util/Utf8.hpp"

namespace ctkm::table {
namespace {

constexpr const char* kLogger = "ctkm.table.reconstruct";

std::string trim(const std::string& text) {
    const char* whitespace = " \t\r\n\f\v";
    const auto begin = text.find_first_not_of(whitespace);
    if (begin == std::string::npos) {
        return std::string();
    }
    const auto end = text.find_last_not_of(whitespace);
    return text.substr(begin, end - begin + 1);
}

}  // namespace

TableCell::TableCell(int cellRow, int cellCol, std::string cellText,
                     std::optional<BoundingBox> cellBox, int cellRowSpan, int cellColSpan)
    : row(cellRow),
      col(cellCol),
      rowSpan(std::max(1, cellRowSpan)),
      colSpan(std::max(1, cellColSpan)),
      box(cellBox),
      text(trim(cellText)) {}

bool TableCell::isEmpty() const { return trim(text).empty(); }

bool TableCell::covers(int targetRow, int targetCol) const {
    return row <= targetRow && targetRow < row + rowSpan && col <= targetCol &&
           targetCol < col + colSpan;
}

Table::Table(std::vector<TableCell> cells, int rows, int cols, std::string source,
             double confidence)
    : cells_(std::move(cells)),
      nRows_(rows),
      nCols_(cols),
      source_(std::move(source)),
      confidence_(confidence) {
    // Giống ``__post_init__`` bản Python: mở rộng kích thước lưới theo ô thực tế.
    for (const auto& cell : cells_) {
        nRows_ = std::max(nRows_, cell.row + cell.rowSpan);
        nCols_ = std::max(nCols_, cell.col + cell.colSpan);
    }
}

bool Table::isEmpty() const {
    for (const auto& cell : cells_) {
        if (!cell.isEmpty()) {
            return false;
        }
    }
    return true;
}

const TableCell* Table::cellAt(int row, int col) const {
    for (const auto& cell : cells_) {
        if (cell.covers(row, col)) {
            return &cell;
        }
    }
    return nullptr;
}

std::vector<const TableCell*> Table::rowCells(int row) const {
    std::vector<const TableCell*> found;
    for (const auto& cell : cells_) {
        if (cell.row <= row && row < cell.row + cell.rowSpan) {
            found.push_back(&cell);
        }
    }
    std::stable_sort(found.begin(), found.end(),
                     [](const TableCell* a, const TableCell* b) { return a->col < b->col; });
    return found;
}

std::vector<const TableCell*> Table::columnCells(int col) const {
    std::vector<const TableCell*> found;
    for (const auto& cell : cells_) {
        if (cell.col <= col && col < cell.col + cell.colSpan) {
            found.push_back(&cell);
        }
    }
    std::stable_sort(found.begin(), found.end(),
                     [](const TableCell* a, const TableCell* b) { return a->row < b->row; });
    return found;
}

std::vector<const TableCell*> Table::sortedCells() const {
    std::vector<const TableCell*> found;
    found.reserve(cells_.size());
    for (const auto& cell : cells_) {
        found.push_back(&cell);
    }
    std::stable_sort(found.begin(), found.end(), [](const TableCell* a, const TableCell* b) {
        if (a->row != b->row) {
            return a->row < b->row;
        }
        return a->col < b->col;
    });
    return found;
}

std::vector<std::vector<std::string>> Table::toGrid() const {
    std::vector<std::vector<std::string>> grid(
        static_cast<std::size_t>(std::max(0, nRows_)),
        std::vector<std::string>(static_cast<std::size_t>(std::max(0, nCols_))));
    for (const auto& cell : cells_) {
        for (int row = cell.row; row < cell.row + cell.rowSpan; ++row) {
            for (int col = cell.col; col < cell.col + cell.colSpan; ++col) {
                if (row >= 0 && row < nRows_ && col >= 0 && col < nCols_) {
                    grid[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] =
                        cell.text;
                }
            }
        }
    }
    return grid;
}

std::string Table::render(std::size_t maxWidth) const {
    const auto grid = toGrid();
    if (grid.empty()) {
        return "(bảng rỗng)";
    }
    std::vector<std::size_t> widths(static_cast<std::size_t>(std::max(0, nCols_)), 0);
    for (const auto& row : grid) {
        for (std::size_t index = 0; index < row.size(); ++index) {
            widths[index] = std::min(maxWidth, std::max(widths[index], utf8::length(row[index])));
        }
    }
    std::string separator = "+";
    for (const auto width : widths) {
        separator += std::string(width + 2, '-');
        separator += '+';
    }

    std::string out = separator;
    for (const auto& row : grid) {
        out += '\n';
        out += '|';
        for (std::size_t index = 0; index < row.size(); ++index) {
            std::string shown = row[index];
            const std::size_t length = utf8::length(shown);
            if (length > widths[index]) {
                shown = utf8::substr(shown, 0, widths[index] - 1) + "…";
            }
            const std::size_t padding = widths[index] - std::min(widths[index], utf8::length(shown));
            out += ' ';
            out += shown;
            out += std::string(padding, ' ');
            out += ' ';
            out += '|';
        }
        out += '\n';
        out += separator;
    }
    return out;
}

std::string joinTokens(const std::vector<OCRToken>& tokens) {
    const auto lines = ocr::groupTokensIntoLines(tokens);
    std::string out;
    for (const auto& line : lines) {
        std::string part;
        for (const auto& token : line) {
            if (token.text.empty()) {
                continue;
            }
            if (!part.empty()) {
                part += ' ';
            }
            part += token.text;
        }
        if (part.empty()) {
            continue;
        }
        if (!out.empty()) {
            out += ' ';
        }
        out += part;
    }
    return trim(out);
}

int assignTokensToCells(std::vector<TableCell>& cells, const std::vector<OCRToken>& tokens,
                        double minOverlap) {
    std::vector<TableCell*> boxed;
    for (auto& cell : cells) {
        if (cell.box.has_value()) {
            boxed.push_back(&cell);
        }
    }

    int assigned = 0;
    for (const auto& token : tokens) {
        if (token.isEmpty()) {
            continue;
        }
        TableCell* best = nullptr;
        double bestScore = 0.0;
        for (auto* cell : boxed) {
            const double score = token.box.overlapRatio(*cell->box);
            if (score > bestScore) {
                bestScore = score;
                best = cell;
            }
        }
        if (best == nullptr || bestScore < minOverlap) {
            log::debug(kLogger, "Token '" + token.text + "' không khớp ô nào");
            continue;
        }
        best->tokens.push_back(token);
        ++assigned;
    }

    for (auto* cell : boxed) {
        if (!cell->tokens.empty()) {
            cell->text = joinTokens(cell->tokens);
        }
    }
    return assigned;
}

std::vector<std::pair<double, double>> mergeIntervals(
    const std::vector<std::pair<double, double>>& intervals, double minGap) {
    if (intervals.empty()) {
        return {};
    }
    std::vector<std::pair<double, double>> ordered = intervals;
    std::stable_sort(ordered.begin(), ordered.end(),
                     [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                         return a.first < b.first;
                     });
    std::vector<std::pair<double, double>> merged{ordered.front()};
    for (std::size_t index = 1; index < ordered.size(); ++index) {
        auto& current = merged.back();
        if (ordered[index].first <= current.second + minGap) {
            current.second = std::max(current.second, ordered[index].second);
        } else {
            merged.push_back(ordered[index]);
        }
    }
    return merged;
}

std::vector<std::pair<double, double>> detectColumnBands(const std::vector<OCRToken>& tokens) {
    if (tokens.empty()) {
        return {};
    }
    double xMin = tokens.front().box.x1();
    double xMax = tokens.front().box.x2();
    for (const auto& token : tokens) {
        xMin = std::min(xMin, token.box.x1());
        xMax = std::max(xMax, token.box.x2());
    }
    const double tableWidth = std::max(1.0, xMax - xMin);

    // Token quá rộng (tiêu đề, ô merge ngang) làm nhoè biên cột nên bị loại ra
    // khi dựng band, nhưng vẫn được gán vào cột ở bước sau.
    std::vector<OCRToken> narrow;
    for (const auto& token : tokens) {
        if (token.box.width() <= tableWidth * kWideTokenRatio) {
            narrow.push_back(token);
        }
    }
    const std::vector<OCRToken>& source = narrow.empty() ? tokens : narrow;

    std::vector<double> heights;
    heights.reserve(source.size());
    for (const auto& token : source) {
        heights.push_back(token.box.height());
    }
    std::sort(heights.begin(), heights.end());
    const double medianHeight = heights.empty() ? 10.0 : heights[heights.size() / 2];
    // Khoảng trắng giữa 2 cột thường lớn hơn khoảng cách giữa 2 từ cùng dòng.
    const double minGap = std::max(6.0, medianHeight * 0.9);

    std::vector<std::pair<double, double>> intervals;
    intervals.reserve(source.size());
    for (const auto& token : source) {
        intervals.emplace_back(token.box.x1(), token.box.x2());
    }
    return mergeIntervals(intervals, minGap);
}

int bestBand(const BoundingBox& box, const std::vector<std::pair<double, double>>& bands) {
    int bestIndex = 0;
    double bestOverlap = -1.0;
    for (std::size_t index = 0; index < bands.size(); ++index) {
        const double overlap =
            std::min(box.x2(), bands[index].second) - std::max(box.x1(), bands[index].first);
        if (overlap > bestOverlap) {
            bestOverlap = overlap;
            bestIndex = static_cast<int>(index);
        }
    }
    return bestIndex;
}

std::optional<Table> clusterTokensToTable(const std::vector<OCRToken>& tokens,
                                          double lineOverlapThreshold, int maxColumns) {
    std::vector<OCRToken> usable;
    for (const auto& token : tokens) {
        if (!token.isEmpty()) {
            usable.push_back(token);
        }
    }
    if (usable.size() < 2) {
        log::warn(kLogger, "Không đủ token để cluster thành bảng (" +
                               std::to_string(usable.size()) + " token)");
        return std::nullopt;
    }

    const auto rows = ocr::groupTokensIntoLines(usable, lineOverlapThreshold);
    if (rows.size() < 2) {
        log::warn(kLogger,
                  "Chỉ gom được " + std::to_string(rows.size()) + " hàng, không dựng được bảng");
        return std::nullopt;
    }

    auto bands = detectColumnBands(usable);
    if (bands.empty()) {
        log::warn(kLogger, "Không xác định được biên cột từ bounding box");
        return std::nullopt;
    }
    if (static_cast<int>(bands.size()) > maxColumns) {
        log::warn(kLogger, "Phát hiện " + std::to_string(bands.size()) + " cột (>" +
                               std::to_string(maxColumns) +
                               ") - có thể do OCR tách từ quá vụn, vẫn tiếp tục");
        bands.resize(static_cast<std::size_t>(maxColumns));
    }

    std::map<std::pair<int, int>, std::vector<OCRToken>> grouped;
    for (std::size_t rowIndex = 0; rowIndex < rows.size(); ++rowIndex) {
        for (const auto& token : rows[rowIndex]) {
            const int colIndex = bestBand(token.box, bands);
            grouped[{static_cast<int>(rowIndex), colIndex}].push_back(token);
        }
    }

    std::vector<TableCell> cells;
    for (const auto& entry : grouped) {
        BoundingBox box = entry.second.front().box;
        for (std::size_t index = 1; index < entry.second.size(); ++index) {
            box = box.merge(entry.second[index].box);
        }
        TableCell cell(entry.first.first, entry.first.second, joinTokens(entry.second), box);
        cell.tokens = entry.second;
        cells.push_back(std::move(cell));
    }
    if (cells.empty()) {
        return std::nullopt;
    }

    Table table(std::move(cells), static_cast<int>(rows.size()),
                static_cast<int>(bands.size()), "cluster", 0.5);
    log::info(kLogger, "Dựng bảng bằng cluster: " + std::to_string(table.rowCount()) +
                           " hàng x " + std::to_string(table.columnCount()) + " cột");
    return table;
}

Table tableFromRows(const std::vector<std::vector<std::string>>& rows,
                    const std::string& source) {
    std::vector<TableCell> cells;
    std::size_t maxColumns = 0;
    for (std::size_t rowIndex = 0; rowIndex < rows.size(); ++rowIndex) {
        maxColumns = std::max(maxColumns, rows[rowIndex].size());
        for (std::size_t colIndex = 0; colIndex < rows[rowIndex].size(); ++colIndex) {
            cells.emplace_back(static_cast<int>(rowIndex), static_cast<int>(colIndex),
                               rows[rowIndex][colIndex]);
        }
    }
    return Table(std::move(cells), static_cast<int>(rows.size()), static_cast<int>(maxColumns),
                 source, 1.0);
}

}  // namespace ctkm::table
