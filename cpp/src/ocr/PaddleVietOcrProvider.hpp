// Provider mặc định: PP-OCRv5 detect (DB) + VietOCR recognize.
//
// Port của ``ocr/paddle_vietocr_provider.py``, giữ nguyên thứ tự xử lý:
//   tiền xử lý ảnh -> detect polygon -> crop + nắn phối cảnh -> VietOCR -> token.
//
// Việc thiếu model (``models/det.onnx``, ``models/vietocr.onnx``) tương ứng với
// ``ImportError`` của bản Python: constructor ném :class:`ProviderUnavailableError`
// và tầng factory tự chuyển sang :class:`TesseractOCR`, không crash.
#pragma once

#include <memory>
#include <string>

#include "ocr/OCRProvider.hpp"
#include "ocr/OnnxDetector.hpp"
#include "ocr/OnnxVietOCR.hpp"
#include "ocr/Preprocess.hpp"

namespace ctkm::ocr {

/// Tham số khởi tạo provider mặc định.
struct PaddleVietOcrOptions {
    std::string detectorModelPath = "models/det.onnx";
    std::string recognizerModelPath = "models/vietocr.onnx";
    std::string vocabularyPath;  // rỗng = suy từ đường dẫn model
    double minConfidence = 0.0;
    PreprocessConfig preprocess;
    DetectorConfig detector;
    RecognizerConfig recognizer;
};

/// Detect bằng PP-OCRv5 (DB), recognize bằng VietOCR.
class PaddleVietOcrProvider : public OCRProvider {
public:
    /// Nạp cả hai model; ném :class:`ProviderUnavailableError` nếu thiếu.
    explicit PaddleVietOcrProvider(PaddleVietOcrOptions options = PaddleVietOcrOptions());
    ~PaddleVietOcrProvider() override;

    std::string name() const override { return "paddle_vietocr"; }

    /// True nếu build có ONNXRuntime và cả hai file model đều tồn tại.
    static bool isAvailable(const PaddleVietOcrOptions& options = PaddleVietOcrOptions());

    OCRResult extract(const std::string& imagePath) override;
    bool setPreprocess(const PreprocessConfig& config) override {
        options_.preprocess = config;
        return true;
    }

private:
    /// Cắt vùng text và nắn phối cảnh về hình chữ nhật.
    cv::Mat crop(const cv::Mat& image, const std::vector<Point>& polygon) const;

    PaddleVietOcrOptions options_;
    std::unique_ptr<OnnxDetector> detector_;
    std::unique_ptr<OnnxVietOCR> recognizer_;
};

}  // namespace ctkm::ocr
