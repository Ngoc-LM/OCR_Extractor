#include "table/Morphology.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <filesystem>
#include <map>

#include "util/Log.hpp"

namespace ctkm::table {
namespace {

constexpr const char* kLogger = "ctkm.table.morphology";

/// Nhị phân hoá thích nghi: chữ/đường kẻ tối -> trắng (255) trên nền đen.
cv::Mat binarize(const cv::Mat& gray) {
    cv::Mat binary;
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY_INV, 25, 15);
    return binary;
}

/// Tách mặt nạ đường kẻ ngang và dọc bằng morphological opening.
///
/// Ý tưởng: 1 kernel hình chữ nhật RẤT dài theo 1 trục chỉ "sống sót" qua phép
/// opening nếu tồn tại 1 đoạn liên tục đủ dài theo trục đó - đúng đặc điểm của
/// đường kẻ bảng, khác với nét chữ (ngắn, đứt đoạn).
void extractLines(const cv::Mat& binary, cv::Mat& horizontal, cv::Mat& vertical) {
    const int width = binary.cols;
    const int height = binary.rows;
    const int horizontalLength =
        std::max(15, static_cast<int>(width * kMinLineLengthRatio));
    const int verticalLength = std::max(15, static_cast<int>(height * kMinLineLengthRatio));

    const cv::Mat horizontalKernel =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(horizontalLength, 1));
    const cv::Mat verticalKernel =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, verticalLength));

    cv::morphologyEx(binary, horizontal, cv::MORPH_OPEN, horizontalKernel, cv::Point(-1, -1), 1);
    cv::morphologyEx(binary, vertical, cv::MORPH_OPEN, verticalKernel, cv::Point(-1, -1), 1);
}

/// Toạ độ trung tâm của từng đường kẻ, suy ra từ hình chiếu mật độ pixel.
///
/// ``axis == 1`` chiếu theo hàng (tìm đường NGANG, mỗi đường ứng với 1 giá trị y);
/// ``axis == 0`` chiếu theo cột (tìm đường DỌC, mỗi đường ứng với 1 giá trị x).
std::vector<double> linePositions(const cv::Mat& mask, int axis) {
    cv::Mat projection;
    // numpy sum(axis=1) = tổng theo từng hàng  -> cv::reduce dim = 1
    // numpy sum(axis=0) = tổng theo từng cột   -> cv::reduce dim = 0
    cv::reduce(mask, projection, axis == 1 ? 1 : 0, cv::REDUCE_SUM, CV_64F);

    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(projection.total()));
    for (int index = 0; index < static_cast<int>(projection.total()); ++index) {
        values.push_back(projection.at<double>(axis == 1 ? index : 0, axis == 1 ? 0 : index));
    }
    if (values.empty()) {
        return {};
    }
    const double peak = *std::max_element(values.begin(), values.end());
    if (peak <= 0.0) {
        return {};
    }
    const double threshold = peak * 0.3;

    std::vector<int> positions;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (values[index] > threshold) {
            positions.push_back(static_cast<int>(index));
        }
    }
    if (positions.empty()) {
        return {};
    }

    std::vector<std::vector<int>> groups{{positions.front()}};
    for (std::size_t index = 1; index < positions.size(); ++index) {
        if (positions[index] - groups.back().back() <= kLineMergeGap) {
            groups.back().push_back(positions[index]);
        } else {
            groups.push_back({positions[index]});
        }
    }

    std::vector<double> centers;
    centers.reserve(groups.size());
    for (const auto& group : groups) {
        double total = 0.0;
        for (const int value : group) {
            total += value;
        }
        centers.push_back(total / static_cast<double>(group.size()));
    }
    return centers;
}

}  // namespace

Grid detectGrid(const cv::Mat& image) {
    if (image.empty()) {
        throw TableMorphologyUnavailableError("Ảnh rỗng, không dò được lưới bảng");
    }
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = image;
    }

    const cv::Mat binary = binarize(gray);
    cv::Mat horizontalMask;
    cv::Mat verticalMask;
    extractLines(binary, horizontalMask, verticalMask);

    Grid grid;
    grid.rowLines = linePositions(horizontalMask, 1);
    grid.colLines = linePositions(verticalMask, 0);
    std::sort(grid.rowLines.begin(), grid.rowLines.end());
    std::sort(grid.colLines.begin(), grid.colLines.end());

    // Phạm vi y của từng đường dọc - dùng để nhận ra bảng lồng nhau.
    for (const double x : grid.colLines) {
        const int column = static_cast<int>(std::round(x));
        const int left = std::max(0, column - 3);
        const int right = std::min(verticalMask.cols, column + 4);
        if (right <= left) {
            continue;
        }
        const cv::Mat window = verticalMask.colRange(left, right);
        cv::Mat rowMax;
        // REDUCE_MAX yêu cầu dtype cùng độ sâu với đầu vào (CV_8U), không phải CV_32S.
        cv::reduce(window, rowMax, 1, cv::REDUCE_MAX);
        int first = -1;
        int last = -1;
        for (int y = 0; y < rowMax.rows; ++y) {
            if (rowMax.at<unsigned char>(y, 0) > 0) {
                if (first < 0) {
                    first = y;
                }
                last = y;
            }
        }
        if (first >= 0) {
            grid.colSegments.push_back(
                ColumnSegment{x, static_cast<double>(first), static_cast<double>(last)});
        }
    }
    return grid;
}

Grid detectGrid(const std::string& imagePath) {
    if (!std::filesystem::is_regular_file(imagePath)) {
        throw TableMorphologyUnavailableError("Không đọc được ảnh: " + imagePath);
    }
    const cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw TableMorphologyUnavailableError("Không đọc được ảnh: " + imagePath);
    }
    return detectGrid(image);
}

std::optional<std::pair<double, double>> selectDensestBand(
    const std::vector<double>& rowLines, const std::vector<ColumnSegment>& colSegments) {
    if (rowLines.size() < 2 || colSegments.empty()) {
        return std::nullopt;
    }

    std::vector<int> counts;
    counts.reserve(rowLines.size() - 1);
    for (std::size_t index = 0; index + 1 < rowLines.size(); ++index) {
        const double top = rowLines[index];
        const double bottom = rowLines[index + 1];
        int count = 0;
        for (const auto& segment : colSegments) {
            if (segment.yStart <= top + kBandCoverageTolerance &&
                segment.yEnd >= bottom - kBandCoverageTolerance) {
                ++count;
            }
        }
        counts.push_back(count);
    }

    const int best = *std::max_element(counts.begin(), counts.end());
    const int worst = *std::min_element(counts.begin(), counts.end());
    if (best == worst) {
        return std::nullopt;  // lưới đồng nhất - chỉ có một bảng
    }

    // Dải liên tiếp dài nhất đạt mức dày nhất.
    std::size_t bestStart = 0;
    std::size_t bestLength = 0;
    std::size_t currentStart = 0;
    std::size_t currentLength = 0;
    for (std::size_t index = 0; index < counts.size(); ++index) {
        if (counts[index] == best) {
            if (currentLength == 0) {
                currentStart = index;
            }
            ++currentLength;
            if (currentLength > bestLength) {
                bestStart = currentStart;
                bestLength = currentLength;
            }
        } else {
            currentLength = 0;
        }
    }
    return std::make_pair(rowLines[bestStart], rowLines[bestStart + bestLength]);
}

std::vector<double> columnsInBand(const std::vector<ColumnSegment>& colSegments, double top,
                                  double bottom) {
    // Chỉ giữ đường kẻ THUỘC VỀ dải này: đường của bảng bao ngoài cũng cắt qua
    // dải, nhưng kéo dài vượt xa hai đầu nên bị loại bằng biên độ dưới đây.
    const double margin = std::max(kBandCoverageTolerance, (bottom - top) * 0.5);
    std::vector<double> columns;
    for (const auto& segment : colSegments) {
        if (segment.yStart <= top + kBandCoverageTolerance &&
            segment.yEnd >= bottom - kBandCoverageTolerance && segment.yStart >= top - margin &&
            segment.yEnd <= bottom + margin) {
            columns.push_back(segment.x);
        }
    }
    std::sort(columns.begin(), columns.end());
    return columns;
}

std::optional<double> trailingRowLimit(const std::vector<double>& rowLines) {
    if (rowLines.size() < 2) {
        return std::nullopt;
    }
    std::vector<double> gaps;
    gaps.reserve(rowLines.size() - 1);
    for (std::size_t index = 0; index + 1 < rowLines.size(); ++index) {
        gaps.push_back(rowLines[index + 1] - rowLines[index]);
    }
    std::sort(gaps.begin(), gaps.end());
    const double medianGap = gaps[gaps.size() / 2];
    if (medianGap <= 0.0) {
        return std::nullopt;
    }
    return rowLines.back() + medianGap;
}

std::optional<int> rowForToken(const BoundingBox& box, const std::vector<double>& rowLines,
                               std::optional<double> trailingLimit) {
    const double centerY = box.center().y;
    for (std::size_t index = 0; index + 1 < rowLines.size(); ++index) {
        if (rowLines[index] <= centerY && centerY < rowLines[index + 1]) {
            return static_cast<int>(index);
        }
    }
    if (trailingLimit.has_value() && !rowLines.empty() && rowLines.back() <= centerY &&
        centerY <= *trailingLimit) {
        return static_cast<int>(rowLines.size()) - 1;
    }
    return std::nullopt;
}

int bestBandNearest(const BoundingBox& box,
                    const std::vector<std::pair<double, double>>& bands) {
    int bestIndex = 0;
    double bestOverlap = 0.0;
    bool foundOverlap = false;
    for (std::size_t index = 0; index < bands.size(); ++index) {
        const double overlap =
            std::min(box.x2(), bands[index].second) - std::max(box.x1(), bands[index].first);
        if (overlap > 0.0 && (!foundOverlap || overlap > bestOverlap)) {
            foundOverlap = true;
            bestOverlap = overlap;
            bestIndex = static_cast<int>(index);
        }
    }
    if (foundOverlap) {
        return bestIndex;
    }

    const double centerX = box.center().x;
    int nearestIndex = 0;
    double nearestDistance = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < bands.size(); ++index) {
        double distance = 0.0;
        if (centerX < bands[index].first) {
            distance = bands[index].first - centerX;
        } else if (centerX > bands[index].second) {
            distance = centerX - bands[index].second;
        }
        if (distance < nearestDistance) {
            nearestDistance = distance;
            nearestIndex = static_cast<int>(index);
        }
    }
    return nearestIndex;
}

Table MorphologyTableRecognizer::recognize(const cv::Mat& image,
                                           const std::vector<OCRToken>& tokens) const {
    const Grid grid = detectGrid(image);
    if (static_cast<int>(grid.rowLines.size()) < kMinLines) {
        throw TableMorphologyUnavailableError(
            "Không đủ đường kẻ ngang để coi là bảng viền rõ (hàng=" +
            std::to_string(grid.rowLines.size()) + ", cần >= " + std::to_string(kMinLines) +
            ")");
    }

    std::vector<OCRToken> usable;
    for (const auto& token : tokens) {
        if (!token.isEmpty()) {
            usable.push_back(token);
        }
    }

    // BẢNG LỒNG BẢNG: giữ lại đúng dải hàng có nhiều đường kẻ dọc cắt qua nhất.
    // Nếu không làm bước này, hàng 0 của lưới là header của bảng NGOÀI, nên biên
    // cột suy từ token hàng 0 sẽ là cột của bảng ngoài chứ không phải bảng dữ
    // liệu bên trong.
    std::vector<double> rowLines = grid.rowLines;
    const auto band = selectDensestBand(rowLines, grid.colSegments);
    if (band.has_value()) {
        const double top = band->first;
        const double bottom = band->second;
        std::vector<double> selected;
        for (const double line : rowLines) {
            if (line >= top - 1.0 && line <= bottom + 1.0) {
                selected.push_back(line);
            }
        }
        std::vector<OCRToken> inside;
        for (const auto& token : usable) {
            const double centerY = token.box.center().y;
            if (centerY >= top && centerY <= bottom) {
                inside.push_back(token);
            }
        }
        if (static_cast<int>(selected.size()) >= kMinLines && !inside.empty()) {
            log::info(kLogger, "Phát hiện bảng lồng nhau: thu hẹp về dải y " +
                                   std::to_string(static_cast<int>(top)) + "-" +
                                   std::to_string(static_cast<int>(bottom)) + " (" +
                                   std::to_string(selected.size()) + "/" +
                                   std::to_string(rowLines.size()) + " đường ngang, " +
                                   std::to_string(inside.size()) + "/" +
                                   std::to_string(usable.size()) + " token)");
            rowLines = selected;
            usable = inside;
        }
    }

    const std::optional<double> limit = trailingRowLimit(rowLines);
    std::vector<std::optional<int>> rowOf(usable.size());
    for (std::size_t index = 0; index < usable.size(); ++index) {
        rowOf[index] = rowForToken(usable[index].box, rowLines, limit);
    }

    // Dò biên cột CHỈ từ token ở HÀNG HEADER (hàng 0), không gộp toàn bộ token
    // mọi hàng: token ở hàng dữ liệu có thể rộng hơn cột chứa nó (VD "MP 20p đầu
    // tiên" tràn ra ngoài cột hẹp "TK thoại"), làm 2 band kề nhau bị nối nhầm
    // thành 1 nếu tính gộp. Thực nghiệm trên ảnh scan thật: gộp toàn bộ token ra
    // 6/8 cột (sai), chỉ dùng hàng header ra đúng 8/8 cột.
    // Ưu tiên ĐƯỜNG KẺ DỌC thật làm biên cột; chỉ suy từ token khi bảng không có
    // kẻ dọc. Trên BM.12, biên suy từ token gộp nhầm "Phí đăng ký" với "TK thoại"
    // thành một cột, còn đường kẻ dọc cho đúng 8 cột.
    std::vector<std::pair<double, double>> bands;
    const std::vector<double> vertical =
        columnsInBand(grid.colSegments, rowLines.front(), rowLines.back());
    if (static_cast<int>(vertical.size()) >= kMinLines) {
        for (std::size_t index = 0; index + 1 < vertical.size(); ++index) {
            bands.emplace_back(vertical[index], vertical[index + 1]);
        }
        // Loại token nằm ngoài bề ngang của bảng (VD chữ của bảng bao ngoài).
        std::vector<OCRToken> inside;
        for (const auto& token : usable) {
            const double centerX = token.box.center().x;
            if (centerX >= vertical.front() - kBandCoverageTolerance &&
                centerX <= vertical.back() + kBandCoverageTolerance) {
                inside.push_back(token);
            }
        }
        usable = inside;
        rowOf.assign(usable.size(), std::nullopt);
        for (std::size_t index = 0; index < usable.size(); ++index) {
            rowOf[index] = rowForToken(usable[index].box, rowLines, limit);
        }
    } else {
        std::vector<OCRToken> headerTokens;
        for (std::size_t index = 0; index < usable.size(); ++index) {
            if (rowOf[index].has_value() && *rowOf[index] == 0) {
                headerTokens.push_back(usable[index]);
            }
        }
        bands = detectColumnBands(headerTokens.empty() ? usable : headerTokens);
    }
    if (bands.empty()) {
        throw TableMorphologyUnavailableError("Không tách được cột nào từ token OCR");
    }

    std::map<std::pair<int, int>, std::vector<OCRToken>> grouped;
    int unmatched = 0;
    for (std::size_t index = 0; index < usable.size(); ++index) {
        if (!rowOf[index].has_value()) {
            ++unmatched;
            continue;
        }
        const int colIndex = bestBandNearest(usable[index].box, bands);
        grouped[{*rowOf[index], colIndex}].push_back(usable[index]);
    }

    if (grouped.empty()) {
        throw TableMorphologyUnavailableError(
            "Không gán được token nào vào lưới (hàng, cột) đã dò");
    }

    std::vector<TableCell> cells;
    std::size_t assigned = 0;
    for (const auto& entry : grouped) {
        BoundingBox box = entry.second.front().box;
        for (std::size_t index = 1; index < entry.second.size(); ++index) {
            box = box.merge(entry.second[index].box);
        }
        TableCell cell(entry.first.first, entry.first.second, joinTokens(entry.second), box);
        cell.tokens = entry.second;
        cells.push_back(std::move(cell));
        assigned += entry.second.size();
    }

    Table table(std::move(cells), static_cast<int>(rowLines.size()) - 1,
                static_cast<int>(bands.size()), kName, 0.8);
    log::info(kLogger,
              "Dựng bảng bằng morphology (hàng theo đường kẻ, cột theo token): " +
                  std::to_string(table.rowCount()) + " hàng x " +
                  std::to_string(table.columnCount()) + " cột, gán " +
                  std::to_string(assigned) + "/" + std::to_string(usable.size()) +
                  " token (" + std::to_string(unmatched) + " token không khớp hàng nào)");
    return table;
}

Table MorphologyTableRecognizer::recognize(const std::string& imagePath,
                                           const std::vector<OCRToken>& tokens) const {
    if (!std::filesystem::is_regular_file(imagePath)) {
        throw TableMorphologyUnavailableError("Không đọc được ảnh: " + imagePath);
    }
    const cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw TableMorphologyUnavailableError("Không đọc được ảnh: " + imagePath);
    }
    return recognize(image, tokens);
}

}  // namespace ctkm::table
