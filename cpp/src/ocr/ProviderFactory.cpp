#include "ocr/ProviderFactory.hpp"

#include <algorithm>
#include <cctype>

#include "util/Log.hpp"

namespace ctkm::ocr {

const std::vector<std::string> kEnginePriority = {"paddle_vietocr", "tesseract"};

namespace {

constexpr const char* kLogger = "ctkm.ocr";

std::string toLower(const std::string& text) {
    std::string out = text;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

std::string trim(const std::string& text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return std::string();
    }
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

std::unique_ptr<OCRProvider> build(const std::string& engine, const ProviderOptions& options) {
    if (engine == "paddle_vietocr") {
        return std::make_unique<PaddleVietOcrProvider>(options.paddleVietOcr);
    }
    if (engine == "tesseract") {
        return std::make_unique<TesseractOCR>(options.tesseract);
    }
    throw ProviderUnavailableError("Engine không hợp lệ: " + engine);
}

}  // namespace

std::vector<std::string> availableEngines(const ProviderOptions& options) {
    std::vector<std::string> result;
    for (const auto& name : kEnginePriority) {
        if (name == "paddle_vietocr" && PaddleVietOcrProvider::isAvailable(options.paddleVietOcr)) {
            result.push_back(name);
        } else if (name == "tesseract" && TesseractOCR::isAvailable()) {
            result.push_back(name);
        }
    }
    return result;
}

std::unique_ptr<OCRProvider> createProvider(const std::string& engine, bool strict,
                                            const ProviderOptions& options) {
    const std::string requested = toLower(trim(engine)).empty() ? std::string(kAutoEngine)
                                                                : toLower(trim(engine));

    std::vector<std::string> candidates;
    if (requested == kAutoEngine) {
        candidates = kEnginePriority;
    } else if (std::find(kEnginePriority.begin(), kEnginePriority.end(), requested) !=
               kEnginePriority.end()) {
        // Thử engine được yêu cầu trước, rồi tới các engine còn lại.
        candidates.push_back(requested);
        for (const auto& name : kEnginePriority) {
            if (name != requested) {
                candidates.push_back(name);
            }
        }
    } else {
        throw ProviderUnavailableError("Engine không hợp lệ: " + engine);
    }

    std::vector<std::string> errors;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        const std::string& name = candidates[index];
        try {
            auto provider = build(name, options);
            if (index > 0) {
                log::warn(kLogger, "Đã fallback từ '" + requested + "' sang provider '" + name +
                                       "'");
            } else {
                log::info(kLogger, "Dùng OCR provider '" + name + "'");
            }
            return provider;
        } catch (const ProviderUnavailableError& error) {
            errors.emplace_back(name + ": " + error.what());
            if (strict && index == 0) {
                throw;
            }
            log::warn(kLogger,
                      "Provider " + name + " không khả dụng (" + error.what() + ")");
        } catch (const std::exception& error) {
            errors.emplace_back(name + ": " + error.what());
            if (strict && index == 0) {
                throw ProviderUnavailableError("Khởi tạo provider " + name + " thất bại: " +
                                               error.what());
            }
            log::warn(kLogger, "Khởi tạo provider " + name + " thất bại (" + error.what() + ")");
        }
    }

    std::string detail;
    for (std::size_t index = 0; index < errors.size(); ++index) {
        if (index > 0) {
            detail += " | ";
        }
        detail += errors[index];
    }
    throw ProviderUnavailableError("Không OCR provider nào khả dụng. Chi tiết: " + detail);
}

}  // namespace ctkm::ocr
