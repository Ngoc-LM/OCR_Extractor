// Port của ``ctkm_extractor/table/__init__.py``: điều phối 4 mức fallback.
//
// 1. Morphology (ưu tiên nhất): dò đường kẻ bảng bằng CV cổ điển.
// 2. PP-Structure: model học sâu (stretch goal, chưa hiện thực - xem PPStructure.hpp).
// 3. Cluster bounding box thủ công: gom theo y rồi theo x.
// 4. Raw text blob: không dựng bảng, để tầng trích xuất dùng regex.
//
// Nhờ vậy pipeline không bao giờ crash khi OCR hoặc table detection sai.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "ocr/OCRProvider.hpp"
#include "table/Morphology.hpp"
#include "table/PPStructure.hpp"
#include "table/Reconstruct.hpp"

namespace ctkm::table {

extern const char* const kStrategyMorphology;
extern const char* const kStrategyPpStructure;
extern const char* const kStrategyCluster;
extern const char* const kStrategyRawText;

/// Kết quả dựng bảng kèm chiến lược đã dùng và cảnh báo phát sinh.
struct TableBuildResult {
    std::optional<Table> table;
    std::string strategy = kStrategyRawText;
    std::string rawText;
    std::vector<std::string> warnings;

    bool hasTable() const { return table.has_value() && !table->isEmpty(); }
};

/// Tuỳ chọn khi dựng bảng (tương ứng các tham số keyword của bản Python).
struct BuildOptions {
    bool useMorphology = true;
    bool usePpStructure = true;
    /// Ảnh đã nạp sẵn; nếu rỗng thì dùng ``OCRResult::imagePath``.
    cv::Mat image;
};

/// Dựng bảng từ kết quả OCR, tự động hạ cấp chiến lược khi thất bại.
TableBuildResult buildTable(const ocr::OCRResult& ocrResult,
                            const BuildOptions& options = BuildOptions());

}  // namespace ctkm::table
