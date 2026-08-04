// Recognizer VietOCR (kiến trúc vgg_transformer) chạy qua ONNXRuntime C++ API.
//
// Port phần *recognition* của ``ocr/paddle_vietocr_provider.py``. Recognizer đa
// ngôn ngữ mặc định của PaddleOCR KHÔNG được dùng: mỗi vùng do detector cắt ra
// đều đi qua VietOCR, vì độ chính xác dấu tiếng Việt cao hơn rõ rệt.
//
// Hợp đồng model (xem README, mục "Model contract"):
//   * Input ảnh: float32 NCHW [1, 3, 32, W], giá trị chia 255 -> [0, 1] (đúng
//     như ``vietocr.tool.translate.process_image``, KHÔNG chuẩn hoá về [-1, 1]).
//   * Model 1 input  -> output [1, T, V] hoặc [T, 1, V]: giải mã greedy 1 lượt.
//   * Model 2 input  -> input thứ hai là chuỗi id đã sinh (int64 [1, L]):
//     giải mã tự hồi quy, mỗi bước lấy argmax của bước cuối.
//   * Bảng ký tự nạp từ ``<model>.vocab`` hoặc ``models/vietocr_vocab.txt``,
//     mỗi dòng một ký tự, 4 dòng đầu là <pad> <sos> <eos> <unk>.
#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "ocr/OCRProvider.hpp"

namespace ctkm::ocr {

/// Tham số tiền xử lý + giải mã của VietOCR.
struct RecognizerConfig {
    /// Chiều cao chuẩn hoá của ảnh đầu vào (VietOCR dùng 32).
    int imageHeight = 32;
    /// Giới hạn chiều rộng sau khi giữ tỉ lệ.
    int minWidth = 32;
    int maxWidth = 512;
    /// Số bước giải mã tối đa khi chạy tự hồi quy.
    int maxLength = 128;
    /// Id đặc biệt trong bảng ký tự của VietOCR.
    int padId = 0;
    int sosId = 1;
    int eosId = 2;
};

/// Kết quả nhận dạng một vùng crop.
struct RecognitionResult {
    std::string text;
    double confidence = 0.0;
};

/// Bọc ``Ort::Session`` chạy model recognition.
class OnnxVietOCR {
public:
    /// Nạp model + bảng ký tự; ném :class:`ProviderUnavailableError` khi thiếu.
    explicit OnnxVietOCR(const std::string& modelPath, const std::string& vocabPath = "",
                         const RecognizerConfig& config = RecognizerConfig());
    ~OnnxVietOCR();

    OnnxVietOCR(const OnnxVietOCR&) = delete;
    OnnxVietOCR& operator=(const OnnxVietOCR&) = delete;
    OnnxVietOCR(OnnxVietOCR&&) noexcept;
    OnnxVietOCR& operator=(OnnxVietOCR&&) noexcept;

    /// True nếu bản build này có ONNXRuntime.
    static bool isSupported();

    /// Nhận dạng một vùng crop BGR; trả text rỗng + confidence 0 khi thất bại.
    RecognitionResult recognize(const cv::Mat& crop);

    /// Số ký tự trong bảng vocab đã nạp.
    std::size_t vocabularySize() const { return vocabulary_.size(); }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    RecognizerConfig config_;
    std::vector<std::string> vocabulary_;
};

/// Nạp bảng ký tự: mỗi dòng một ký tự. Ném :class:`ProviderUnavailableError`.
std::vector<std::string> loadVocabulary(const std::string& vocabPath);

}  // namespace ctkm::ocr
