// Port của ``ctkm_extractor/ocr/__init__.py``: chọn provider theo tên và tự
// fallback khi provider được yêu cầu không khả dụng.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ocr/OCRProvider.hpp"
#include "ocr/PaddleVietOcrProvider.hpp"
#include "ocr/TesseractOCR.hpp"

namespace ctkm::ocr {

/// Tên engine hợp lệ cho CLI, theo thứ tự ưu tiên khi chạy "auto".
extern const std::vector<std::string> kEnginePriority;

constexpr const char* kAutoEngine = "auto";

/// Tham số dùng chung khi dựng provider.
struct ProviderOptions {
    PaddleVietOcrOptions paddleVietOcr;
    TesseractOptions tesseract;
};

/// Danh sách engine thực sự chạy được trong môi trường hiện tại.
std::vector<std::string> availableEngines(const ProviderOptions& options = ProviderOptions());

/// Tạo OCR provider theo tên, tự fallback khi thiếu model/thư viện.
///
/// @param engine "paddle_vietocr" (mặc định), "tesseract", hoặc "auto".
/// @param strict true thì không fallback - lỗi được ném ra ngoài.
/// @throws ProviderUnavailableError khi không provider nào dùng được.
std::unique_ptr<OCRProvider> createProvider(const std::string& engine = "paddle_vietocr",
                                            bool strict = false,
                                            const ProviderOptions& options = ProviderOptions());

}  // namespace ctkm::ocr
