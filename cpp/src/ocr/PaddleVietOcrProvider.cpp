#include "ocr/PaddleVietOcrProvider.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>

#include "util/Log.hpp"

namespace ctkm::ocr {
namespace {

constexpr const char* kLogger = "ctkm.ocr.paddle_vietocr";

/// Số pixel nới thêm quanh mỗi vùng crop, giúp recognizer không bị cụt dấu.
constexpr int kCropPadding = 2;

double distance(const cv::Point2f& a, const cv::Point2f& b) {
    return std::hypot(static_cast<double>(a.x - b.x), static_cast<double>(a.y - b.y));
}

/// Sắp 4 điểm theo thứ tự trên-trái, trên-phải, dưới-phải, dưới-trái.
std::array<cv::Point2f, 4> orderPoints(const std::vector<Point>& polygon) {
    std::vector<cv::Point2f> points;
    points.reserve(polygon.size());
    for (const auto& point : polygon) {
        points.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
    }
    std::array<cv::Point2f, 4> ordered{};
    double minSum = 0.0;
    double maxSum = 0.0;
    double minDiff = 0.0;
    double maxDiff = 0.0;
    for (std::size_t index = 0; index < points.size(); ++index) {
        const double sum = points[index].x + points[index].y;
        const double diff = points[index].y - points[index].x;
        if (index == 0 || sum < minSum) {
            minSum = sum;
            ordered[0] = points[index];
        }
        if (index == 0 || sum > maxSum) {
            maxSum = sum;
            ordered[2] = points[index];
        }
        if (index == 0 || diff < minDiff) {
            minDiff = diff;
            ordered[1] = points[index];
        }
        if (index == 0 || diff > maxDiff) {
            maxDiff = diff;
            ordered[3] = points[index];
        }
    }
    return ordered;
}

}  // namespace

bool PaddleVietOcrProvider::isAvailable(const PaddleVietOcrOptions& options) {
    if (!OnnxDetector::isSupported() || !OnnxVietOCR::isSupported()) {
        return false;
    }
    return std::filesystem::is_regular_file(options.detectorModelPath) &&
           std::filesystem::is_regular_file(options.recognizerModelPath);
}

PaddleVietOcrProvider::PaddleVietOcrProvider(PaddleVietOcrOptions options)
    : options_(std::move(options)) {
    if (!std::filesystem::is_regular_file(options_.detectorModelPath)) {
        throw ProviderUnavailableError("Thiếu model detect: " + options_.detectorModelPath);
    }
    if (!std::filesystem::is_regular_file(options_.recognizerModelPath)) {
        throw ProviderUnavailableError("Thiếu model VietOCR: " +
                                       options_.recognizerModelPath);
    }
    detector_ = std::make_unique<OnnxDetector>(options_.detectorModelPath, options_.detector);
    recognizer_ = std::make_unique<OnnxVietOCR>(options_.recognizerModelPath,
                                                options_.vocabularyPath, options_.recognizer);
}

PaddleVietOcrProvider::~PaddleVietOcrProvider() = default;

cv::Mat PaddleVietOcrProvider::crop(const cv::Mat& image, const std::vector<Point>& polygon) const {
    try {
        std::array<cv::Point2f, 4> source = orderPoints(polygon);
        int width = static_cast<int>(std::max(distance(source[0], source[1]),
                                              distance(source[3], source[2])));
        int height = static_cast<int>(std::max(distance(source[0], source[3]),
                                               distance(source[1], source[2])));
        width += 2 * kCropPadding;
        height += 2 * kCropPadding;
        if (width < 4 || height < 4) {
            return cv::Mat();
        }
        const std::array<cv::Point2f, 4> destination{
            cv::Point2f(0.0F, 0.0F), cv::Point2f(static_cast<float>(width - 1), 0.0F),
            cv::Point2f(static_cast<float>(width - 1), static_cast<float>(height - 1)),
            cv::Point2f(0.0F, static_cast<float>(height - 1))};

        const cv::Mat matrix = cv::getPerspectiveTransform(source.data(), destination.data());
        cv::Mat warped;
        cv::warpPerspective(image, warped, matrix, cv::Size(width, height), cv::INTER_LINEAR,
                            cv::BORDER_REPLICATE);
        // Ảnh dọc (chữ xoay 90 độ) thì xoay lại cho recognizer.
        if (height > width * 1.5) {
            cv::Mat rotated;
            cv::rotate(warped, rotated, cv::ROTATE_90_CLOCKWISE);
            return rotated;
        }
        return warped;
    } catch (const std::exception& error) {
        log::warn(kLogger, std::string("Crop vùng text thất bại: ") + error.what());
        return cv::Mat();
    }
}

OCRResult PaddleVietOcrProvider::extract(const std::string& imagePath) {
    OCRResult result;
    result.provider = name();
    result.imagePath = imagePath;

    PreprocessResult prepared = preprocessImage(imagePath, options_.preprocess);
    if (!prepared.processed) {
        result.warnings.emplace_back("Không tiền xử lý được ảnh, dùng ảnh gốc");
    }
    cv::Mat image = prepared.image;
    if (image.empty()) {
        throw ProviderUnavailableError("Không đọc được ảnh: " + imagePath);
    }
    result.imageSize = std::make_pair(image.cols, image.rows);
    result.processedImage = image;

    const std::vector<std::vector<Point>> polygons = detector_->detect(image);
    if (polygons.empty()) {
        result.warnings.emplace_back("Detector không tìm thấy vùng text nào");
        log::warn(kLogger, "PP-OCR detector không tìm thấy vùng text trong " + imagePath);
    }

    for (const auto& polygon : polygons) {
        const cv::Mat region = crop(image, polygon);
        if (region.empty()) {
            continue;
        }
        const RecognitionResult recognition = recognizer_->recognize(region);
        if (recognition.text.empty()) {
            continue;
        }
        if (recognition.confidence < options_.minConfidence) {
            log::debug(kLogger, "Bỏ token confidence thấp: " + recognition.text);
            continue;
        }
        OCRToken token(recognition.text, BoundingBox::fromPolygon(polygon),
                       recognition.confidence);
        token.polygon = polygon;
        if (!token.isEmpty()) {
            result.tokens.push_back(std::move(token));
        }
    }
    return result;
}

}  // namespace ctkm::ocr
