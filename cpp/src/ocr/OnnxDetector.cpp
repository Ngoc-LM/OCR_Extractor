#include "ocr/OnnxDetector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>

#include "util/Log.hpp"

#ifdef CTKM_WITH_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace ctkm::ocr {
namespace {

constexpr const char* kLogger = "ctkm.ocr.detector";

/// Chuẩn hoá ảnh theo đúng cấu hình PP-OCR: BGR->RGB, /255, (x-mean)/std, NCHW.
[[maybe_unused]] std::vector<float> normalizeToNchw(const cv::Mat& bgr) {
    static const float kMean[3] = {0.485F, 0.456F, 0.406F};
    static const float kStd[3] = {0.229F, 0.224F, 0.225F};

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat asFloat;
    rgb.convertTo(asFloat, CV_32FC3, 1.0 / 255.0);

    const int height = asFloat.rows;
    const int width = asFloat.cols;
    std::vector<float> tensor(static_cast<std::size_t>(3) * height * width);
    for (int y = 0; y < height; ++y) {
        const auto* row = asFloat.ptr<cv::Vec3f>(y);
        for (int x = 0; x < width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                const float value = (row[x][channel] - kMean[channel]) / kStd[channel];
                tensor[(static_cast<std::size_t>(channel) * height + y) * width + x] = value;
            }
        }
    }
    return tensor;
}

/// Resize giữ tỉ lệ, cạnh dài nhất <= limit, hai cạnh làm tròn về bội số của 32.
[[maybe_unused]] cv::Mat resizeForDetector(const cv::Mat& image, int limitSideLength,
                                           double& scaleX, double& scaleY) {
    const int height = image.rows;
    const int width = image.cols;
    double ratio = 1.0;
    const int longest = std::max(height, width);
    if (longest > limitSideLength) {
        ratio = static_cast<double>(limitSideLength) / static_cast<double>(longest);
    }
    int targetHeight = static_cast<int>(std::round(height * ratio));
    int targetWidth = static_cast<int>(std::round(width * ratio));
    targetHeight = std::max(32, (targetHeight / 32) * 32);
    targetWidth = std::max(32, (targetWidth / 32) * 32);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(targetWidth, targetHeight));
    scaleX = static_cast<double>(width) / static_cast<double>(targetWidth);
    scaleY = static_cast<double>(height) / static_cast<double>(targetHeight);
    return resized;
}

/// 4 đỉnh của rotated rect, sắp theo thứ tự trên-trái, trên-phải, dưới-phải, dưới-trái.
std::vector<cv::Point2f> orderedRectPoints(const cv::RotatedRect& rect) {
    cv::Point2f raw[4];
    rect.points(raw);
    std::vector<cv::Point2f> points(raw, raw + 4);
    std::sort(points.begin(), points.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
    // 2 điểm trái và 2 điểm phải, mỗi cặp sắp theo y.
    std::vector<cv::Point2f> left{points[0], points[1]};
    std::vector<cv::Point2f> right{points[2], points[3]};
    std::sort(left.begin(), left.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
    std::sort(right.begin(), right.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
    return {left[0], right[0], right[1], left[1]};
}

/// Điểm trung bình của bản đồ xác suất bên trong contour.
float boxScore(const cv::Mat& probabilityMap, const std::vector<cv::Point>& contour) {
    cv::Rect bounds = cv::boundingRect(contour) & cv::Rect(0, 0, probabilityMap.cols,
                                                           probabilityMap.rows);
    if (bounds.width <= 0 || bounds.height <= 0) {
        return 0.0F;
    }
    cv::Mat mask = cv::Mat::zeros(bounds.size(), CV_8UC1);
    std::vector<cv::Point> shifted;
    shifted.reserve(contour.size());
    for (const auto& point : contour) {
        shifted.emplace_back(point.x - bounds.x, point.y - bounds.y);
    }
    std::vector<std::vector<cv::Point>> polygons{shifted};
    cv::fillPoly(mask, polygons, cv::Scalar(255));
    return static_cast<float>(cv::mean(probabilityMap(bounds), mask)[0]);
}

/// Nới rộng rotated rect ra ngoài theo hệ số unclip của DB.
cv::RotatedRect unclip(const cv::RotatedRect& rect, float ratio) {
    const float width = rect.size.width;
    const float height = rect.size.height;
    const float area = width * height;
    const float perimeter = 2.0F * (width + height);
    if (perimeter <= 0.0F) {
        return rect;
    }
    // Xấp xỉ phép offset polygon của DB: khoảng nới = area * ratio / perimeter,
    // áp dụng cho cả hai cạnh (mỗi chiều nới 2 lần khoảng đó).
    const float distance = area * ratio / perimeter;
    cv::RotatedRect expanded = rect;
    expanded.size.width = width + 2.0F * distance;
    expanded.size.height = height + 2.0F * distance;
    return expanded;
}

}  // namespace

std::vector<std::vector<Point>> decodeDbBoxes(const cv::Mat& probabilityMap,
                                              const DetectorConfig& config, double scaleX,
                                              double scaleY, int originalWidth,
                                              int originalHeight) {
    std::vector<std::vector<Point>> polygons;
    if (probabilityMap.empty()) {
        return polygons;
    }

    cv::Mat binary;
    cv::threshold(probabilityMap, binary, config.binaryThreshold, 255.0, cv::THRESH_BINARY);
    cv::Mat binaryU8;
    binary.convertTo(binaryU8, CV_8UC1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryU8, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    const int limit = std::min(static_cast<int>(contours.size()), config.maxCandidates);
    for (int index = 0; index < limit; ++index) {
        const auto& contour = contours[static_cast<std::size_t>(index)];
        if (contour.size() < 4) {
            continue;
        }
        cv::RotatedRect rect = cv::minAreaRect(contour);
        if (std::min(rect.size.width, rect.size.height) < config.minBoxSide) {
            continue;
        }
        if (boxScore(probabilityMap, contour) < config.boxThreshold) {
            continue;
        }
        rect = unclip(rect, config.unclipRatio);
        if (std::min(rect.size.width, rect.size.height) < config.minBoxSide + 2.0F) {
            continue;
        }

        std::vector<Point> polygon;
        polygon.reserve(4);
        for (const auto& point : orderedRectPoints(rect)) {
            const double x = std::clamp(static_cast<double>(point.x) * scaleX, 0.0,
                                        static_cast<double>(originalWidth - 1));
            const double y = std::clamp(static_cast<double>(point.y) * scaleY, 0.0,
                                        static_cast<double>(originalHeight - 1));
            polygon.push_back(Point{x, y});
        }
        polygons.push_back(std::move(polygon));
    }
    return polygons;
}

#ifdef CTKM_WITH_ONNXRUNTIME

struct OnnxDetector::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ctkm-detector"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::string inputName;
    std::string outputName;
};

bool OnnxDetector::isSupported() { return true; }

OnnxDetector::OnnxDetector(const std::string& modelPath, const DetectorConfig& config)
    : impl_(std::make_unique<Impl>()), config_(config) {
    if (!std::filesystem::is_regular_file(modelPath)) {
        throw ProviderUnavailableError("Không tìm thấy model detect: " + modelPath);
    }
    try {
        impl_->options.SetIntraOpNumThreads(1);
        impl_->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        impl_->session =
            std::make_unique<Ort::Session>(impl_->env, modelPath.c_str(), impl_->options);
        const auto input = impl_->session->GetInputNameAllocated(0, impl_->allocator);
        const auto output = impl_->session->GetOutputNameAllocated(0, impl_->allocator);
        impl_->inputName = input.get();
        impl_->outputName = output.get();
    } catch (const std::exception& error) {
        throw ProviderUnavailableError(std::string("Không nạp được model detect: ") +
                                       error.what());
    }
}

OnnxDetector::~OnnxDetector() = default;
OnnxDetector::OnnxDetector(OnnxDetector&&) noexcept = default;
OnnxDetector& OnnxDetector::operator=(OnnxDetector&&) noexcept = default;

std::vector<std::vector<Point>> OnnxDetector::detect(const cv::Mat& image) {
    if (image.empty()) {
        return {};
    }
    double scaleX = 1.0;
    double scaleY = 1.0;
    const cv::Mat resized = resizeForDetector(image, config_.limitSideLength, scaleX, scaleY);
    std::vector<float> tensor = normalizeToNchw(resized);

    const std::array<int64_t, 4> shape{1, 3, resized.rows, resized.cols};
    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, tensor.data(), tensor.size(), shape.data(), shape.size());

    const char* inputNames[] = {impl_->inputName.c_str()};
    const char* outputNames[] = {impl_->outputName.c_str()};

    std::vector<Ort::Value> outputs;
    try {
        outputs = impl_->session->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1,
                                      outputNames, 1);
    } catch (const std::exception& error) {
        log::warn(kLogger, std::string("Detector chạy lỗi: ") + error.what());
        return {};
    }
    if (outputs.empty()) {
        log::warn(kLogger, "Detector không trả về output nào");
        return {};
    }

    const auto info = outputs.front().GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> outputShape = info.GetShape();
    if (outputShape.size() < 2) {
        log::warn(kLogger, "Shape output của detector không hợp lệ");
        return {};
    }
    const int mapHeight = static_cast<int>(outputShape[outputShape.size() - 2]);
    const int mapWidth = static_cast<int>(outputShape[outputShape.size() - 1]);
    const float* data = outputs.front().GetTensorData<float>();
    const cv::Mat probabilityMap(mapHeight, mapWidth, CV_32FC1, const_cast<float*>(data));

    // Bản đồ xác suất có thể nhỏ hơn ảnh đã resize; quy đổi về ảnh gốc.
    const double mapScaleX = scaleX * static_cast<double>(resized.cols) /
                             static_cast<double>(std::max(1, mapWidth));
    const double mapScaleY = scaleY * static_cast<double>(resized.rows) /
                             static_cast<double>(std::max(1, mapHeight));

    return decodeDbBoxes(probabilityMap.clone(), config_, mapScaleX, mapScaleY, image.cols,
                         image.rows);
}

#else  // CTKM_WITH_ONNXRUNTIME

struct OnnxDetector::Impl {};

bool OnnxDetector::isSupported() { return false; }

OnnxDetector::OnnxDetector(const std::string& modelPath, const DetectorConfig& config)
    : impl_(nullptr), config_(config) {
    (void)modelPath;
    throw ProviderUnavailableError(
        "Bản build này không có ONNXRuntime - cấu hình lại CMake với "
        "-DONNXRUNTIME_ROOT_DIR=<đường dẫn> để dùng provider paddle_vietocr");
}

OnnxDetector::~OnnxDetector() = default;
OnnxDetector::OnnxDetector(OnnxDetector&&) noexcept = default;
OnnxDetector& OnnxDetector::operator=(OnnxDetector&&) noexcept = default;

std::vector<std::vector<Point>> OnnxDetector::detect(const cv::Mat& image) {
    (void)image;
    return {};
}

#endif  // CTKM_WITH_ONNXRUNTIME

}  // namespace ctkm::ocr
