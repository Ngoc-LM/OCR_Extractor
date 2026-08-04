// Port của ``ctkm_extractor/ocr/base.py``.
//
// Mọi provider (ONNX detector + VietOCR, Tesseract, ...) đều trả về cùng một cấu
// trúc :class:`OCRResult` gồm danh sách token ``text + bounding box +
// confidence``. Nhờ vậy tầng table và tầng extraction không phụ thuộc engine OCR
// cụ thể.
#pragma once

#include <opencv2/core.hpp>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>

#include <utility>
#include <vector>

namespace ctkm::ocr {

// Khai báo trước thay vì include Preprocess.hpp: file đó đã include ngược lại
// OCRProvider.hpp nên include hai chiều sẽ vỡ. Tham chiếu tới kiểu chưa hoàn
// chỉnh là hợp lệ trong khai báo hàm.
struct PreprocessConfig;

/// Lỗi chung của tầng OCR.
class OCRError : public std::runtime_error {
public:
    explicit OCRError(const std::string& message) : std::runtime_error(message) {}
};

/// Provider không dùng được (thiếu model, thiếu thư viện, thiếu gói ngôn ngữ).
class ProviderUnavailableError : public OCRError {
public:
    explicit ProviderUnavailableError(const std::string& message) : OCRError(message) {}
};

struct Point {
    double x = 0.0;
    double y = 0.0;
};

/// Hộp bao (axis-aligned) của một token text, đơn vị pixel.
class BoundingBox {
public:
    BoundingBox() = default;

    BoundingBox(double x1, double y1, double x2, double y2) {
        // Chuẩn hoá để x1 <= x2 và y1 <= y2, tránh box "âm" do OCR trả ngược toạ độ.
        x1_ = std::min(x1, x2);
        x2_ = std::max(x1, x2);
        y1_ = std::min(y1, y2);
        y2_ = std::max(y1, y2);
    }

    /// Tạo box từ polygon (4 điểm trở lên) do detector trả về.
    static BoundingBox fromPolygon(const std::vector<Point>& points);
    /// Tạo box từ định dạng ``(x, y, width, height)`` (Tesseract dùng dạng này).
    static BoundingBox fromXywh(double x, double y, double width, double height) {
        return BoundingBox(x, y, x + width, y + height);
    }

    double x1() const { return x1_; }
    double y1() const { return y1_; }
    double x2() const { return x2_; }
    double y2() const { return y2_; }

    double width() const { return x2_ - x1_; }
    double height() const { return y2_ - y1_; }
    double area() const { return std::max(0.0, width()) * std::max(0.0, height()); }
    Point center() const { return Point{(x1_ + x2_) / 2.0, (y1_ + y2_) / 2.0}; }

    /// Diện tích giao nhau giữa hai box (0 nếu rời nhau).
    double intersectionArea(const BoundingBox& other) const;
    /// Tỉ lệ diện tích của ``*this`` nằm trong ``other`` (0..1).
    double overlapRatio(const BoundingBox& other) const;
    /// Intersection-over-Union, dùng khi so khớp ô bảng với token.
    double iou(const BoundingBox& other) const;
    /// True nếu tâm của ``other`` nằm trong ``*this``.
    bool containsCenterOf(const BoundingBox& other) const;
    /// Tỉ lệ chồng lấn theo trục y so với chiều cao nhỏ hơn (dùng gom dòng).
    double verticalOverlapRatio(const BoundingBox& other) const;
    /// Hộp bao nhỏ nhất chứa cả hai box.
    BoundingBox merge(const BoundingBox& other) const;

private:
    double x1_ = 0.0;
    double y1_ = 0.0;
    double x2_ = 0.0;
    double y2_ = 0.0;
};

/// Một đoạn text đã nhận dạng kèm vị trí và độ tin cậy.
struct OCRToken {
    std::string text;
    BoundingBox box;
    double confidence = 1.0;
    std::vector<Point> polygon;

    OCRToken() = default;
    OCRToken(std::string tokenText, BoundingBox tokenBox, double tokenConfidence = 1.0);

    bool isEmpty() const { return text.empty(); }
};

/// Kết quả OCR của một ảnh.
struct OCRResult {
    std::vector<OCRToken> tokens;
    std::optional<std::string> imagePath;
    std::optional<std::pair<int, int>> imageSize;  // (width, height)
    std::string provider = "unknown";
    std::vector<std::string> warnings;
    /// Ảnh ĐÃ TIỀN XỬ LÝ mà provider dùng để sinh token (BGR), rỗng nếu không có.
    ///
    /// Bắt buộc phải mang theo vì bounding box của token nằm trong hệ toạ độ của
    /// ảnh này: tiền xử lý có thể phóng to (``minWidth``) và xoay (deskew) ảnh.
    /// Tầng dựng bảng phải dò đường kẻ trên đúng ảnh đó, đọc lại ảnh gốc từ đĩa sẽ
    /// lệch hệ toạ độ - và trên ảnh scan bị nghiêng thì không dò được đường kẻ
    /// ngang nào, khiến morphology (mức dựng bảng tốt nhất) bị vô hiệu.
    cv::Mat processedImage;

    /// Token còn text sau khi trim.
    std::vector<OCRToken> nonEmptyTokens() const;
    /// Toàn bộ text gom theo dòng (trên xuống dưới, trái sang phải).
    std::string rawText() const;
    /// Confidence trung bình của các token không rỗng.
    double meanConfidence() const;

    /// Dựng ``OCRResult`` giả lập từ text thuần - dùng cho test và cho ``--text-file``.
    static OCRResult fromText(const std::string& text, const std::string& provider = "text");
};

/// Sắp xếp token theo thứ tự đọc (y trước, x sau).
std::vector<OCRToken> sortTokensReadingOrder(const std::vector<OCRToken>& tokens);

/// Gom token thành các dòng dựa trên độ chồng lấn theo trục y.
std::vector<std::vector<OCRToken>> groupTokensIntoLines(const std::vector<OCRToken>& tokens,
                                                        double overlapThreshold = 0.4);

/// Interface mà mọi engine OCR phải hiện thực.
class OCRProvider {
public:
    virtual ~OCRProvider() = default;

    /// Tên ngắn dùng cho CLI (``--engine``) và log.
    virtual std::string name() const = 0;
    /// Chạy OCR trên ảnh và trả về danh sách token có toạ độ.
    virtual OCRResult extract(const std::string& imagePath) = 0;
    /// Đổi cấu hình tiền xử lý; trả về false nếu provider không có bước này.
    ///
    /// Dùng cho chế độ tự chọn nhị phân hoá: đổi cấu hình rẻ hơn nhiều so với
    /// dựng lại provider (dựng lại sẽ nạp lại model recognition).
    virtual bool setPreprocess(const PreprocessConfig&) { return false; }
};

}  // namespace ctkm::ocr
