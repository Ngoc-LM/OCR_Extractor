// Text detector PP-OCRv5 (thuật toán DB - Differentiable Binarization) chạy qua
// ONNXRuntime C++ API.
//
// Port phần *detection* của ``ocr/paddle_vietocr_provider.py``: bản Python gọi
// ``PaddleOCR(det=True, rec=False)``; bản C++ nạp thẳng model đã export ONNX nên
// không cần Paddle Inference SDK. Đầu ra giống hệt: danh sách polygon 4 điểm.
//
// Nếu build không có ONNXRuntime (``CTKM_WITH_ONNXRUNTIME`` không bật), lớp này
// vẫn tồn tại nhưng constructor ném :class:`ProviderUnavailableError` để tầng
// factory tự fallback sang Tesseract - đúng hành vi ``try/except ImportError``
// của bản Python.
#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "ocr/OCRProvider.hpp"

namespace ctkm::ocr {

/// Tham số hậu xử lý DB, mặc định trùng cấu hình PP-OCR.
struct DetectorConfig {
    /// Cạnh dài nhất của ảnh sau khi resize (chuẩn PP-OCR: 960).
    int limitSideLength = 960;
    /// Ngưỡng nhị phân hoá bản đồ xác suất.
    float binaryThreshold = 0.3F;
    /// Ngưỡng điểm trung bình của một box mới được giữ lại.
    float boxThreshold = 0.6F;
    /// Hệ số nới rộng polygon sau khi lấy minAreaRect.
    float unclipRatio = 1.5F;
    /// Cạnh nhỏ nhất (pixel) của box được chấp nhận.
    float minBoxSide = 3.0F;
    /// Số box tối đa lấy ra từ một ảnh.
    int maxCandidates = 1000;
};

/// Bọc ``Ort::Session`` chạy model detect và hậu xử lý DB.
class OnnxDetector {
public:
    /// Nạp model; ném :class:`ProviderUnavailableError` nếu thiếu file/thiếu ORT.
    explicit OnnxDetector(const std::string& modelPath,
                          const DetectorConfig& config = DetectorConfig());
    ~OnnxDetector();

    OnnxDetector(const OnnxDetector&) = delete;
    OnnxDetector& operator=(const OnnxDetector&) = delete;
    OnnxDetector(OnnxDetector&&) noexcept;
    OnnxDetector& operator=(OnnxDetector&&) noexcept;

    /// True nếu bản build này có ONNXRuntime.
    static bool isSupported();

    /// Trả về danh sách polygon 4 điểm (toạ độ trên ảnh gốc truyền vào).
    std::vector<std::vector<Point>> detect(const cv::Mat& image);

    const DetectorConfig& config() const { return config_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    DetectorConfig config_;
};

/// Hậu xử lý DB tách riêng để test được độc lập với ONNXRuntime.
/// ``probabilityMap`` là ảnh CV_32F cùng kích thước ảnh đã resize.
std::vector<std::vector<Point>> decodeDbBoxes(const cv::Mat& probabilityMap,
                                              const DetectorConfig& config, double scaleX,
                                              double scaleY, int originalWidth,
                                              int originalHeight);

}  // namespace ctkm::ocr
