#include "table/TableBuilder.hpp"

#include <opencv2/imgcodecs.hpp>

#include "util/Log.hpp"

namespace ctkm::table {

const char* const kStrategyMorphology = "morphology";
const char* const kStrategyPpStructure = "pp_structure";
const char* const kStrategyCluster = "cluster";
const char* const kStrategyRawText = "raw_text";

namespace {
constexpr const char* kLogger = "ctkm.table";
}

TableBuildResult buildTable(const ocr::OCRResult& ocrResult, const BuildOptions& options) {
    TableBuildResult result;
    result.rawText = ocrResult.rawText();
    const std::vector<ocr::OCRToken> tokens = ocrResult.nonEmptyTokens();

    if (tokens.empty()) {
        const std::string message = "OCR không trả về token nào - chuyển sang xử lý raw text";
        log::warn(kLogger, message);
        result.warnings.push_back(message);
        result.strategy = kStrategyRawText;
        return result;
    }

    // Nguồn ảnh cho morphology/PP-Structure: ảnh truyền vào, nếu không có thì
    // nạp lại từ đường dẫn trong kết quả OCR.
    // Ưu tiên ảnh ĐÃ TIỀN XỬ LÝ mà provider dùng để sinh token: bounding box của
    // token nằm trong hệ toạ độ ảnh đó. Đọc lại ảnh gốc từ đĩa sẽ lệch hệ toạ độ
    // khi tiền xử lý có phóng to/xoay ảnh - và với ảnh scan bị nghiêng thì không
    // dò được đường kẻ ngang nào nên morphology bị vô hiệu.
    cv::Mat image = options.image;
    if (image.empty()) {
        image = ocrResult.processedImage;
    }
    if (image.empty() && ocrResult.imagePath.has_value()) {
        image = cv::imread(*ocrResult.imagePath, cv::IMREAD_COLOR);
    }
    const bool hasImage = !image.empty();

    if (options.useMorphology && hasImage) {
        try {
            MorphologyTableRecognizer recognizer;
            if (MorphologyTableRecognizer::isAvailable()) {
                result.table = recognizer.recognize(image, tokens);
                result.strategy = kStrategyMorphology;
                return result;
            }
            result.warnings.emplace_back("Thiếu OpenCV cho morphology - fallback PP-Structure");
        } catch (const TableMorphologyUnavailableError& error) {
            const std::string message =
                std::string("Morphology không dựng được bảng (") + error.what() +
                ") - fallback PP-Structure";
            log::info(kLogger, message);
            result.warnings.push_back(message);
        } catch (const std::exception& error) {
            const std::string message = std::string("Morphology lỗi bất ngờ (") + error.what() +
                                        ") - fallback PP-Structure";
            log::warn(kLogger, message);
            result.warnings.push_back(message);
        }
    } else if (options.useMorphology) {
        result.warnings.emplace_back("Không có ảnh nguồn cho morphology - fallback PP-Structure");
    }

    if (options.usePpStructure && hasImage) {
        try {
            PPStructureTableRecognizer recognizer;
            result.table = recognizer.recognize(image, tokens);
            result.strategy = kStrategyPpStructure;
            return result;
        } catch (const TableStructureUnavailableError& error) {
            const std::string message = std::string("PP-Structure không dùng được (") +
                                        error.what() + ") - fallback cluster bounding box";
            log::warn(kLogger, message);
            result.warnings.push_back(message);
        } catch (const std::exception& error) {
            const std::string message = std::string("PP-Structure lỗi bất ngờ (") +
                                        error.what() + ") - fallback cluster bounding box";
            log::warn(kLogger, message);
            result.warnings.push_back(message);
        }
    } else if (options.usePpStructure) {
        result.warnings.emplace_back(
            "Không có ảnh nguồn cho PP-Structure - fallback cluster bounding box");
    }

    std::optional<Table> clustered;
    try {
        clustered = clusterTokensToTable(tokens);
    } catch (const std::exception& error) {
        log::warn(kLogger, std::string("Cluster bounding box lỗi: ") + error.what());
        result.warnings.emplace_back(std::string("Cluster bounding box lỗi: ") + error.what());
        clustered = std::nullopt;
    }

    if (clustered.has_value() && !clustered->isEmpty()) {
        result.table = clustered;
        result.strategy = kStrategyCluster;
        return result;
    }

    const std::string message =
        "Không dựng được bảng - xử lý toàn bộ text như một khối duy nhất";
    log::warn(kLogger, message);
    result.warnings.push_back(message);
    result.strategy = kStrategyRawText;
    result.table.reset();
    return result;
}

}  // namespace ctkm::table
