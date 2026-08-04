// Mức fallback 2 của tầng table: PP-Structure table recognition.
//
// STRETCH GOAL - CHƯA HIỆN THỰC. Bản Python dùng ``paddleocr.PPStructure`` để
// nhận ô gộp bằng model học sâu. Port sang ONNX/C++ tốn effort đáng kể (cần
// export thêm model table-structure, parse HTML cấu trúc, ghép bbox) so với lợi
// ích: bảng CTKM có đường kẻ rõ nên mức 1 (morphology) đã xử lý tốt.
//
// Lớp này giữ đúng interface để chuỗi 4 mức fallback không đổi: ``recognize()``
// luôn ném :class:`TableStructureUnavailableError`, nên ``buildTable`` rơi thẳng
// từ morphology xuống cluster bounding box - vẫn đúng thứ tự hạ cấp của bản
// Python khi PP-Structure không khả dụng.
//
// Phần HTML parser (rowspan/colspan) và hàm ghép token vào ô theo overlap của
// bản Python đã có sẵn tương đương ở ``assignTokensToCells`` trong
// ``table/Reconstruct.hpp``, nên khi bổ sung model chỉ cần điền phần suy cấu
// trúc ô.
#pragma once

#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "ocr/OCRProvider.hpp"
#include "table/Reconstruct.hpp"

namespace ctkm::table {

/// PP-Structure không dùng được hoặc không cho kết quả đáng tin cậy.
class TableStructureUnavailableError : public std::runtime_error {
public:
    explicit TableStructureUnavailableError(const std::string& message)
        : std::runtime_error(message) {}
};

/// Tỉ lệ token được gán vào ô tối thiểu để coi kết quả PP-Structure là đáng tin.
constexpr double kMinAssignRatio = 0.5;

/// Bọc model PP-Structure (chưa hiện thực trong bản C++ - xem ghi chú đầu file).
class PPStructureTableRecognizer {
public:
    static constexpr const char* kName = "pp_structure";

    explicit PPStructureTableRecognizer(std::string modelPath = "models/table_structure.onnx");

    /// Luôn false ở bản C++ hiện tại (stretch goal chưa làm).
    static bool isAvailable();

    /// @throws TableStructureUnavailableError - luôn ném ở bản hiện tại.
    Table recognize(const cv::Mat& image, const std::vector<OCRToken>& tokens) const;
    Table recognize(const std::string& imagePath, const std::vector<OCRToken>& tokens) const;

private:
    std::string modelPath_;
};

}  // namespace ctkm::table
