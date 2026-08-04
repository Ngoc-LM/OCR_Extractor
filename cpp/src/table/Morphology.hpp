// Port của ``ctkm_extractor/table/morphology.py`` - mức dựng bảng ƯU TIÊN NHẤT.
//
// Dò cấu trúc bảng bằng CV cổ điển: morphological opening tách đường kẻ, hàng
// suy từ đường kẻ ngang, cột suy từ đường kẻ dọc của dải hàng đang xét và chỉ
// suy từ toạ độ token OCR của HÀNG HEADER khi bảng không có đủ kẻ dọc. Tất
// định, không cần model, chạy được cả khi dùng OCR provider fallback.
#pragma once

#include <opencv2/core.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ocr/OCRProvider.hpp"
#include "table/Reconstruct.hpp"

namespace ctkm::table {

/// Cần tối thiểu chừng này đường kẻ ngang/dọc mới coi là "có bảng viền rõ".
constexpr int kMinLines = 3;

/// Độ dài tối thiểu của một đoạn được coi là đường kẻ bảng, theo tỉ lệ cạnh ảnh.
///
/// LƯU Ý - vì sao 0.05 chứ không phải 0.35 như bản đầu: kernel dài bằng 0.35 cạnh
/// ảnh chỉ giữ được đường kẻ của bảng LỚN NHẤT trang. Trên biểu mẫu BM.12 thật
/// (2480x3505), bảng CTKM nằm lồng trong một ô của bảng ngoài và chỉ cao ~410px,
/// nên toàn bộ 9 đường kẻ dọc của nó bị phép opening xoá sạch (kernel yêu cầu
/// 1226px liên tục) - dò ra đúng 5 đường dọc của bảng ngoài. Hạ xuống 0.05 thì dò
/// đủ 14 đường dọc (5 ngoài + 9 trong). Nét chữ không sống sót nổi một kernel dài
/// hàng trăm pixel nên không sinh đường kẻ giả.
constexpr double kMinLineLengthRatio = 0.05;

/// Dung sai (pixel) khi xét một đường kẻ dọc có cắt hết một dải hàng hay không.
constexpr double kBandCoverageTolerance = 5.0;

/// Khoảng cách (pixel) giữa 2 vị trí đường kẻ được coi là cùng một đường.
///
/// LƯU Ý: giá trị 15 đã hiệu chỉnh thực nghiệm trên ảnh scan 200 DPI thật - nén
/// JPEG/khử răng cưa khiến 1 đường kẻ đôi khi tách thành 2 đỉnh cách nhau
/// ~12-13px; giá trị nhỏ hơn sẽ tách nhầm 1 đường thành 2.
constexpr int kLineMergeGap = 15;

/// Thiếu OpenCV, không đọc được ảnh, hoặc không dò đủ đường kẻ bảng.
class TableMorphologyUnavailableError : public std::runtime_error {
public:
    explicit TableMorphologyUnavailableError(const std::string& message)
        : std::runtime_error(message) {}
};

/// Một đường kẻ dọc kèm phạm vi y của nó.
///
/// Phạm vi y là thứ phân biệt bảng lồng nhau: đường kẻ của bảng ngoài trải dài
/// toàn bảng, còn đường kẻ của bảng con chỉ nằm trong một dải hàng.
struct ColumnSegment {
    double x = 0.0;
    double yStart = 0.0;
    double yEnd = 0.0;
};

/// Toạ độ các đường kẻ ngang (y) và dọc (x) dò được.
struct Grid {
    std::vector<double> rowLines;
    std::vector<double> colLines;
    std::vector<ColumnSegment> colSegments;
};

/// Chọn dải hàng có NHIỀU ĐƯỜNG KẺ DỌC CẮT QUA NHẤT - tức bảng "dày" nhất.
///
/// Dùng để tách bảng con ra khỏi bảng lồng nhau: trên biểu mẫu BM.12 thật, mọi
/// dải hàng của bảng ngoài chỉ có 5 đường dọc cắt qua, riêng 2 dải chứa bảng CTKM
/// có 14 - chọn đúng 2 dải đó là ra bảng cần trích xuất.
///
/// Trả ``std::nullopt`` khi mọi dải đều có số đường dọc như nhau (ảnh chỉ có một
/// bảng), khi đó tầng gọi giữ nguyên toàn bộ lưới.
std::optional<std::pair<double, double>> selectDensestBand(
    const std::vector<double>& rowLines, const std::vector<ColumnSegment>& colSegments);

/// Toạ độ x của các đường kẻ dọc cắt HẾT dải hàng ``[top, bottom]`` và THUỘC VỀ
/// dải đó (đường của bảng bao ngoài kéo dài vượt xa hai đầu nên bị loại).
///
/// Đây là biên cột chính xác nhất khi bảng có kẻ dọc rõ - chính xác hơn suy từ
/// toạ độ token, vì 2 tiêu đề cột nằm sát nhau dễ bị gộp thành một band.
std::vector<double> columnsInBand(const std::vector<ColumnSegment>& colSegments, double top,
                                  double bottom);

/// Dò toạ độ các đường kẻ ngang/dọc từ ảnh.
/// @throws TableMorphologyUnavailableError khi ảnh không đọc được.
Grid detectGrid(const cv::Mat& image);

/// Nạp ảnh từ đường dẫn rồi dò lưới.
Grid detectGrid(const std::string& imagePath);

/// Giới hạn y của "hàng cuối chưa đóng", bằng 1 bước hàng kể từ đường kẻ cuối.
std::optional<double> trailingRowLimit(const std::vector<double>& rowLines);

/// Chỉ số hàng chứa tâm của ``box``, suy từ các đường kẻ ngang đã dò.
///
/// ``trailingLimit`` (nếu có) là toạ độ y tối đa còn được coi là thuộc hàng cuối
/// CHƯA ĐÓNG VIỀN: ảnh scan/cắt cúp thường thiếu đường kẻ dưới cùng, khi đó
/// token của hàng cuối nằm dưới đường kẻ cuối và sẽ bị mất trắng nếu không có
/// ngoại lệ này. Token nằm phía TRÊN đường kẻ đầu (tiêu đề trang) vẫn bị loại.
std::optional<int> rowForToken(const BoundingBox& box, const std::vector<double>& rowLines,
                               std::optional<double> trailingLimit = std::nullopt);

/// Chỉ số cột của ``box``: ưu tiên cột chồng lấn nhiều nhất theo trục x.
///
/// LƯU Ý (bug thực tế đã sửa): band cột chỉ rộng bằng chữ trong header, nên token
/// hàng dữ liệu dài hơn tiêu đề cột hoàn toàn có thể KHÔNG chồng lấn band nào.
/// Bản đầu khởi tạo ``best_overlap = -1.0`` rồi so ``overlap > best_overlap`` -
/// sai, vì overlap của token nằm ngoài band là số âm lớn nên không bao giờ vượt
/// -1.0, kết quả luôn rơi về cột 0 và dính vào ô nhãn (case thật: ô "Ưu đãi Data"
/// nuốt luôn "tối đa 8gb/1 ngày" khiến dataGB đọc ra 8 thay vì 60). Cách đúng:
/// tìm band có overlap > 0 trước; không có thì chọn band GẦN NHẤT theo khoảng cách.
int bestBandNearest(const BoundingBox& box, const std::vector<std::pair<double, double>>& bands);

/// Nhận diện cấu trúc bảng bằng morphology cổ điển - không cần model.
class MorphologyTableRecognizer {
public:
    static constexpr const char* kName = "morphology";

    /// True nếu bản build có OpenCV (luôn đúng, giữ cho khớp interface bản Python).
    static bool isAvailable() { return true; }

    /// Dựng bảng: hàng theo đường kẻ ảnh, cột theo toạ độ token hàng header.
    /// @throws TableMorphologyUnavailableError khi không đủ đường kẻ ngang,
    ///         không tách được cột, hoặc không token nào khớp hàng nào.
    Table recognize(const cv::Mat& image, const std::vector<OCRToken>& tokens) const;

    /// Nạp ảnh từ đường dẫn rồi dựng bảng.
    Table recognize(const std::string& imagePath, const std::vector<OCRToken>& tokens) const;
};

}  // namespace ctkm::table
