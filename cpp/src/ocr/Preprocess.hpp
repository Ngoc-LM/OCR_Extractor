// Port của ``ctkm_extractor/ocr/preprocess.py``.
//
// Pipeline: grayscale -> deskew -> adaptive threshold. Mục tiêu là giảm nhiễu
// watermark và làm thẳng ảnh chụp nghiêng trước khi đưa vào text detector.
// Áp dụng cho CẢ HAI provider (ONNX và Tesseract) như bản Python.
#pragma once

#include <opencv2/core.hpp>
#include <string>

#include "ocr/OCRProvider.hpp"

namespace ctkm::ocr {

/// Tham số tiền xử lý ảnh (mặc định trùng ``PreprocessConfig`` bản Python).
struct PreprocessConfig {
    bool grayscale = true;
    bool deskew = true;
    bool adaptiveThreshold = true;
    bool denoise = true;
    /// Góc lệch tối đa (độ) còn được coi là hợp lệ để xoay; lớn hơn thì bỏ qua.
    double maxSkewAngle = 15.0;
    /// Kích thước cửa sổ của adaptive threshold (phải lẻ).
    int blockSize = 31;
    /// Hằng số C trừ đi khỏi giá trị trung bình cục bộ.
    int thresholdC = 15;
    /// Phóng to ảnh nhỏ để detector bắt được chữ nhỏ (0 = tắt).
    int minWidth = 1000;
};

/// Đọc ảnh từ đĩa; ném :class:`OCRError` nếu không đọc được.
cv::Mat loadImage(const std::string& imagePath);

/// Chuyển ảnh sang thang xám (idempotent với ảnh đã 1 kênh).
cv::Mat toGrayscale(const cv::Mat& image);

/// Ước lượng góc nghiêng của văn bản (độ, dương = ngược chiều kim đồng hồ).
double estimateSkewAngle(const cv::Mat& gray, double maxAngle = 15.0);

/// Xoay ảnh về phương ngang dựa trên góc nghiêng ước lượng.
cv::Mat deskew(const cv::Mat& image, double maxAngle = 15.0);

/// Nhị phân hoá thích nghi - loại watermark nhạt, giữ nét chữ đậm.
cv::Mat adaptiveThreshold(const cv::Mat& gray, int blockSize = 31, int c = 15);

/// Phóng to ảnh hẹp để detector bắt được dòng chữ nhỏ.
cv::Mat upscaleIfSmall(const cv::Mat& image, int minWidth);

/// Kết quả tiền xử lý: ảnh BGR 3 kênh + cờ pipeline có chạy trọn vẹn không.
struct PreprocessResult {
    cv::Mat image;
    bool processed = false;
};

/// Chạy toàn bộ pipeline tiền xử lý trên một file ảnh.
PreprocessResult preprocessImage(const std::string& imagePath,
                                 const PreprocessConfig& config = PreprocessConfig());

}  // namespace ctkm::ocr
