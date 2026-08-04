#include "ocr/TesseractOCR.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>

#include "util/Log.hpp"

#ifdef CTKM_WITH_TESSERACT
#include <leptonica/allheaders.h>
#include <tesseract/baseapi.h>
#endif

namespace ctkm::ocr {

const std::vector<std::string> kDefaultLanguages = {"vie", "vie+eng", "eng"};
const std::vector<int> kDefaultPsmCandidates = {4, 11, 6};

namespace {
constexpr const char* kLogger = "ctkm.ocr.tesseract";
}

PsmRun selectBestPsmRun(const std::vector<int>& candidates,
                        const std::function<std::vector<OCRToken>(int)>& run,
                        std::vector<std::string>* errors) {
    PsmRun best;
    std::pair<int, double> bestKey{-1, -1.0};

    // Chạy HẾT mọi chế độ phân đoạn - xem ghi chú ở TesseractOCR.hpp về việc vì
    // sao không được dừng sớm khi một psm "trông có vẻ đủ tốt".
    for (const int psm : candidates) {
        std::vector<OCRToken> tokens;
        try {
            tokens = run(psm);
        } catch (const std::exception& error) {
            log::warn(kLogger, "Tesseract lỗi với --psm " + std::to_string(psm) + ": " +
                                   error.what());
            if (errors != nullptr) {
                errors->emplace_back(std::string("psm ") + std::to_string(psm) + ": " +
                                     error.what());
            }
            continue;
        }
        double confidence = 0.0;
        if (!tokens.empty()) {
            for (const auto& token : tokens) {
                confidence += token.confidence;
            }
            confidence /= static_cast<double>(tokens.size());
        }
        const std::pair<int, double> key{static_cast<int>(tokens.size()), confidence};
        if (key > bestKey) {
            bestKey = key;
            best.psm = psm;
            best.tokens = std::move(tokens);
        }
    }
    return best;
}

/// Chọn gói ngôn ngữ tốt nhất đang có; cảnh báo nếu thiếu gói tiếng Việt.
std::string pickLanguage(const std::string& requested,
                         const std::function<bool(const std::string&)>& usable) {
    if (!requested.empty()) {
        if (usable(requested)) {
            return requested;
        }
        log::warn(kLogger, "Tesseract thiếu gói ngôn ngữ " + requested + ", thử dùng mặc định");
    }
    for (const auto& candidate : kDefaultLanguages) {
        if (candidate == requested) {
            continue;
        }
        if (usable(candidate)) {
            if (candidate == "eng") {
                log::warn(kLogger,
                          "Không tìm thấy gói 'vie' - độ chính xác dấu tiếng Việt sẽ rất "
                          "thấp. Cài bằng: apt-get install tesseract-ocr-vie");
            }
            return candidate;
        }
    }
    log::warn(kLogger, "Không xác định được gói ngôn ngữ Tesseract, dùng 'eng'");
    return "eng";
}

#ifdef CTKM_WITH_TESSERACT

namespace {

/// RAII cho ``PIX*`` của Leptonica - không để rò rỉ khi có exception.
class PixGuard {
public:
    explicit PixGuard(PIX* pix = nullptr) : pix_(pix) {}
    ~PixGuard() { reset(); }

    PixGuard(const PixGuard&) = delete;
    PixGuard& operator=(const PixGuard&) = delete;
    PixGuard(PixGuard&& other) noexcept : pix_(other.pix_) { other.pix_ = nullptr; }
    PixGuard& operator=(PixGuard&& other) noexcept {
        if (this != &other) {
            reset();
            pix_ = other.pix_;
            other.pix_ = nullptr;
        }
        return *this;
    }

    void reset(PIX* pix = nullptr) {
        if (pix_ != nullptr) {
            pixDestroy(&pix_);
        }
        pix_ = pix;
    }

    PIX* get() const { return pix_; }
    explicit operator bool() const { return pix_ != nullptr; }

private:
    PIX* pix_ = nullptr;
};

/// Chuyển ``cv::Mat`` sang PIX 8bpp thang xám của Leptonica.
///
/// Dùng 8bpp thay vì 32bpp: PIX 32bpp dựng thủ công có byte alpha = 0 và
/// ``spp`` = 4, khiến Tesseract xử lý ảnh khác hẳn so với ảnh RGB mà bản Python
/// đưa vào (kết quả OCR lệch rõ rệt trên ảnh nhiễu). Tesseract dù sao cũng quy
/// về thang xám trước khi nhị phân hoá.
PIX* matToPix(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
        return nullptr;
    }
    if (gray.depth() != CV_8U) {
        cv::Mat converted;
        gray.convertTo(converted, CV_8U);
        gray = converted;
    }

    PIX* pix = pixCreate(gray.cols, gray.rows, 8);
    if (pix == nullptr) {
        return nullptr;
    }
    l_uint32* data = pixGetData(pix);
    const int wpl = pixGetWpl(pix);
    for (int y = 0; y < gray.rows; ++y) {
        const auto* row = gray.ptr<unsigned char>(y);
        l_uint32* line = data + static_cast<std::ptrdiff_t>(y) * wpl;
        for (int x = 0; x < gray.cols; ++x) {
            SET_DATA_BYTE(line, x, row[x]);
        }
    }
    return pix;
}

/// True nếu Tesseract nạp được gói ngôn ngữ này (kể cả dạng ghép "vie+eng").
///
/// Cố tình KHÔNG dùng ``GetAvailableLanguagesAsVector()``: chữ ký của nó đổi
/// giữa hai đời Tesseract (``GenericVector<STRING>*`` ở bản 4,
/// ``std::vector<std::string>*`` ở bản 5) và ``GenericVector`` chỉ được khai báo
/// trước trong ``baseapi.h``, nên dùng được nó phải kéo theo header nội bộ đã bị
/// gỡ ở bản 5. ``Init()`` thì giống nhau ở mọi đời, và trả lời đúng câu hỏi ta
/// cần: gói này có thực sự dùng được không, chứ không chỉ có mặt trên đĩa.
bool canInitLanguage(const std::string& language) {
    tesseract::TessBaseAPI probe;
    if (probe.Init(nullptr, language.c_str()) != 0) {
        return false;
    }
    probe.End();
    return true;
}


}  // namespace

/// Giữ engine Tesseract; ``End()`` được gọi trong destructor (RAII).
struct TesseractOCR::Impl {
    tesseract::TessBaseAPI api;
    bool initialised = false;

    ~Impl() {
        if (initialised) {
            api.End();
        }
    }
};

bool TesseractOCR::isAvailable() {
    tesseract::TessBaseAPI probe;
    const bool ok = probe.Init(nullptr, "eng") == 0 || probe.Init(nullptr, "vie") == 0;
    if (ok) {
        probe.End();
    }
    return ok;
}

TesseractOCR::TesseractOCR(TesseractOptions options)
    : impl_(std::make_unique<Impl>()), options_(std::move(options)) {
    language_ = pickLanguage(options_.language, canInitLanguage);
    if (impl_->api.Init(nullptr, language_.c_str()) != 0) {
        throw ProviderUnavailableError("Không khởi tạo được Tesseract với ngôn ngữ " +
                                       language_);
    }
    impl_->initialised = true;
}

TesseractOCR::~TesseractOCR() = default;

std::vector<OCRToken> TesseractOCR::runOnce(const cv::Mat& image, int psm) {
    std::vector<OCRToken> tokens;
    PixGuard pix(matToPix(image));
    if (!pix) {
        log::warn(kLogger, "Không chuyển được ảnh sang PIX");
        return tokens;
    }

    impl_->api.SetPageSegMode(static_cast<tesseract::PageSegMode>(psm));
    impl_->api.SetImage(pix.get());
    if (impl_->api.Recognize(nullptr) != 0) {
        log::warn(kLogger, "Tesseract Recognize() trả về lỗi");
        impl_->api.Clear();
        return tokens;
    }

    std::unique_ptr<tesseract::ResultIterator> iterator(impl_->api.GetIterator());
    if (iterator == nullptr) {
        impl_->api.Clear();
        return tokens;
    }
    const auto level = tesseract::RIL_WORD;
    do {
        std::unique_ptr<char[]> word(iterator->GetUTF8Text(level));
        if (word == nullptr) {
            continue;
        }
        std::string text(word.get());
        if (text.find_first_not_of(" \t\r\n\f\v") == std::string::npos) {
            continue;
        }
        double confidence = static_cast<double>(iterator->Confidence(level));
        // Tesseract dùng giá trị âm cho vùng không phải token thật.
        if (confidence < 0.0) {
            continue;
        }
        confidence /= 100.0;
        if (confidence < options_.minConfidence) {
            continue;
        }
        int left = 0;
        int top = 0;
        int right = 0;
        int bottom = 0;
        if (!iterator->BoundingBox(level, &left, &top, &right, &bottom)) {
            continue;
        }
        tokens.emplace_back(text, BoundingBox(left, top, right, bottom), confidence);
    } while (iterator->Next(level));

    impl_->api.Clear();
    // Bỏ token rỗng sau khi trim, giống bản Python.
    tokens.erase(std::remove_if(tokens.begin(), tokens.end(),
                                [](const OCRToken& token) { return token.isEmpty(); }),
                 tokens.end());
    return tokens;
}

OCRResult TesseractOCR::extract(const std::string& imagePath) {
    OCRResult result;
    result.provider = name();
    result.imagePath = imagePath;

    PreprocessResult prepared = preprocessImage(imagePath, options_.preprocess);
    if (!prepared.processed) {
        result.warnings.emplace_back("Không tiền xử lý được ảnh, dùng ảnh gốc");
    }
    const cv::Mat image = prepared.image;
    if (image.empty()) {
        throw ProviderUnavailableError("Không đọc được ảnh: " + imagePath);
    }
    result.imageSize = std::make_pair(image.cols, image.rows);
    result.processedImage = image;

    std::vector<int> candidates;
    if (options_.psm >= 0) {
        candidates.push_back(options_.psm);
    } else {
        candidates = kDefaultPsmCandidates;
    }

    PsmRun best = selectBestPsmRun(
        candidates, [&](int psm) { return runOnce(image, psm); }, &result.warnings);

    if (best.psm < 0) {
        return result;
    }
    log::debug(kLogger, "Tesseract chọn --psm " + std::to_string(best.psm) + " (" +
                            std::to_string(best.tokens.size()) + " token)");
    if (best.tokens.empty()) {
        result.warnings.emplace_back("Tesseract không nhận dạng được text nào");
        log::warn(kLogger, "Tesseract không nhận dạng được text nào trong " + imagePath);
    }
    result.tokens = std::move(best.tokens);
    return result;
}

#else  // CTKM_WITH_TESSERACT

struct TesseractOCR::Impl {};

bool TesseractOCR::isAvailable() { return false; }

TesseractOCR::TesseractOCR(TesseractOptions options)
    : impl_(nullptr), options_(std::move(options)) {
    throw ProviderUnavailableError(
        "Bản build này không có Tesseract - cài libtesseract-dev rồi cấu hình lại CMake");
}

TesseractOCR::~TesseractOCR() = default;

std::vector<OCRToken> TesseractOCR::runOnce(const cv::Mat& image, int psm) {
    (void)image;
    (void)psm;
    return {};
}

OCRResult TesseractOCR::extract(const std::string& imagePath) {
    (void)imagePath;
    throw ProviderUnavailableError("Bản build này không có Tesseract");
}

#endif  // CTKM_WITH_TESSERACT

}  // namespace ctkm::ocr
