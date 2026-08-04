#include "ocr/Preprocess.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <filesystem>

#include "util/Log.hpp"

namespace ctkm::ocr {
namespace {
constexpr const char* kLogger = "ctkm.ocr.preprocess";
}

cv::Mat loadImage(const std::string& imagePath) {
    if (!std::filesystem::is_regular_file(imagePath)) {
        throw OCRError("Không tìm thấy ảnh: " + imagePath);
    }
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw OCRError("Không đọc được ảnh (định dạng không hỗ trợ?): " + imagePath);
    }
    return image;
}

cv::Mat toGrayscale(const cv::Mat& image) {
    if (image.channels() == 1) {
        return image;
    }
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

double estimateSkewAngle(const cv::Mat& gray, double maxAngle) {
    try {
        cv::Mat inverted;
        cv::bitwise_not(gray, inverted);
        cv::Mat binary;
        cv::threshold(inverted, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        std::vector<cv::Point> coordinates;
        cv::findNonZero(binary, coordinates);
        if (coordinates.size() < 50) {
            return 0.0;
        }
        const cv::RotatedRect rect = cv::minAreaRect(coordinates);
        double angle = rect.angle;
        // Quy góc về khoảng (-45, 45]. Cần CẢ HAI nhánh vì các bản OpenCV trả góc
        // theo convention khác nhau: đo trên cùng một ảnh nghiêng 2 độ, OpenCV 4.6
        // trả 2.02 còn OpenCV 5.0 trả -87.98. Thiếu nhánh cộng 90 thì trên OpenCV 5
        // mọi ảnh nghiêng nhẹ đều bị coi là lệch gần 90 độ và bỏ qua deskew.
        if (angle > 45.0) {
            angle -= 90.0;
        } else if (angle < -45.0) {
            angle += 90.0;
        }
        if (std::abs(angle) > maxAngle) {
            log::debug(kLogger, "Bỏ qua deskew: góc ước lượng " + std::to_string(angle) +
                                    " độ vượt ngưỡng");
            return 0.0;
        }
        return angle;
    } catch (const std::exception& error) {
        log::warn(kLogger, std::string("Ước lượng góc nghiêng thất bại: ") + error.what());
        return 0.0;
    }
}

cv::Mat deskew(const cv::Mat& image, double maxAngle) {
    const cv::Mat gray = toGrayscale(image);
    const double angle = estimateSkewAngle(gray, maxAngle);
    if (std::abs(angle) < 0.15) {
        return image;
    }
    const cv::Point2f center(static_cast<float>(image.cols) / 2.0F,
                             static_cast<float>(image.rows) / 2.0F);
    const cv::Mat matrix = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat rotated;
    cv::warpAffine(image, rotated, matrix, image.size(), cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    log::debug(kLogger, "Đã deskew ảnh " + std::to_string(angle) + " độ");
    return rotated;
}

cv::Mat adaptiveThreshold(const cv::Mat& gray, int blockSize, int c) {
    int block = (blockSize % 2 == 1) ? blockSize : blockSize + 1;
    block = std::max(3, block);
    cv::Mat binary;
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,
                          block, c);
    return binary;
}

cv::Mat upscaleIfSmall(const cv::Mat& image, int minWidth) {
    if (minWidth <= 0 || image.cols >= minWidth) {
        return image;
    }
    const double scale = static_cast<double>(minWidth) / static_cast<double>(image.cols);
    cv::Mat resized;
    cv::resize(image, resized,
               cv::Size(static_cast<int>(image.cols * scale), static_cast<int>(image.rows * scale)),
               0, 0, cv::INTER_CUBIC);
    return resized;
}

PreprocessResult preprocessImage(const std::string& imagePath, const PreprocessConfig& config) {
    PreprocessResult result;
    cv::Mat image = loadImage(imagePath);
    try {
        image = upscaleIfSmall(image, config.minWidth);
        if (config.deskew) {
            image = deskew(image, config.maxSkewAngle);
        }
        cv::Mat working = config.grayscale ? toGrayscale(image) : image;
        if (config.grayscale && config.denoise) {
            cv::Mat filtered;
            cv::bilateralFilter(working, filtered, 5, 50, 50);
            working = filtered;
        }
        if (config.grayscale && config.adaptiveThreshold) {
            working = adaptiveThreshold(working, config.blockSize, config.thresholdC);
        }
        if (working.channels() == 1) {
            cv::Mat bgr;
            cv::cvtColor(working, bgr, cv::COLOR_GRAY2BGR);
            working = bgr;
        }
        result.image = working;
        result.processed = true;
        return result;
    } catch (const std::exception& error) {
        log::warn(kLogger,
                  std::string("Tiền xử lý thất bại (") + error.what() + "), dùng ảnh gốc");
        result.image = image;
        result.processed = false;
        return result;
    }
}

}  // namespace ctkm::ocr
