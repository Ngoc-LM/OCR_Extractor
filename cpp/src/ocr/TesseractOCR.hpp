// Provider fallback nhẹ: Tesseract OCR qua API C++ (``tesseract::TessBaseAPI``).
//
// Port của ``ocr/tesseract_ocr.py``, giữ nguyên cả phần chọn ``--psm`` tự động:
// bảng có đường kẻ bị psm 6 (mặc định của Tesseract) cắt sai dòng, nên provider
// thử lần lượt {4, 11, 6} rồi giữ kết quả nhiều token nhất.
//
// Đánh đổi có chủ đích: không cần tải model nặng, nhưng độ chính xác dấu tiếng
// Việt thấp hơn rõ rệt so với provider mặc định.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ocr/OCRProvider.hpp"
#include "ocr/Preprocess.hpp"

namespace ctkm::ocr {

/// Thứ tự thử ngôn ngữ: ưu tiên gói tiếng Việt, sau đó mới tới tiếng Anh.
extern const std::vector<std::string> kDefaultLanguages;

/// Các chế độ phân đoạn trang thử lần lượt khi không chỉ định ``--psm``.
extern const std::vector<int> kDefaultPsmCandidates;

/// Kết quả một lượt chạy Tesseract với một chế độ phân đoạn.
struct PsmRun {
    int psm = -1;
    std::vector<OCRToken> tokens;
};

/// Chạy HẾT các chế độ phân đoạn rồi chọn lượt nhiều token nhất.
///
/// LƯU Ý - KHÔNG dừng sớm ở psm đầu tiên "trông có vẻ ổn": bản đầu có luật dừng
/// sớm (>= 8 token và confidence >= 0.6 thì bỏ qua các psm còn lại). Trên biểu
/// mẫu BM.12 thật (render 300 DPI), psm 4 được thử đầu tiên và đạt ngay ngưỡng
/// đó (80 token, confidence 0.82) nên vòng lặp dừng lại - dù nó KHÔNG đọc được
/// bảng CTKM bên trong. Đo trên cùng ảnh: psm 4 = 79 token, psm 11 = 161 token,
/// psm 6 = 195 token, và chỉ psm 6 đọc ra được "CTKMN180X", "163,636.3636",
/// "150.534.213".
///
/// Tách riêng khỏi :class:`TesseractOCR` để test được mà không cần binary
/// tesseract: ``run`` là callback chạy một lượt, ``errors`` nhận thông báo lỗi.
PsmRun selectBestPsmRun(const std::vector<int>& candidates,
                        const std::function<std::vector<OCRToken>(int)>& run,
                        std::vector<std::string>* errors = nullptr);

/// Chọn gói ngôn ngữ Tesseract tốt nhất đang dùng được.
///
/// ``usable`` trả lời "Tesseract nạp được gói này không". Tách ra thành callback
/// để test được logic chọn mà không cần binary tesseract, và để tránh phụ thuộc
/// ``GetAvailableLanguagesAsVector()`` - hàm đổi chữ ký giữa Tesseract 4 và 5.
std::string pickLanguage(const std::string& requested,
                         const std::function<bool(const std::string&)>& usable);

/// Tham số khởi tạo provider fallback.
struct TesseractOptions {
    /// Ngôn ngữ ép dùng; rỗng = tự chọn theo gói đang cài.
    std::string language;
    /// -1 = thử lần lượt :data:`kDefaultPsmCandidates`.
    int psm = -1;
    int oem = 3;
    double minConfidence = 0.0;
    PreprocessConfig preprocess;
};

/// Nhận dạng bằng Tesseract, lấy cả bounding box từng từ.
class TesseractOCR : public OCRProvider {
public:
    /// Khởi tạo; ném :class:`ProviderUnavailableError` nếu thiếu thư viện/gói ngôn ngữ.
    explicit TesseractOCR(TesseractOptions options = TesseractOptions());
    ~TesseractOCR() override;

    std::string name() const override { return "tesseract"; }

    /// True nếu bản build có Tesseract và khởi tạo được engine.
    static bool isAvailable();

    OCRResult extract(const std::string& imagePath) override;
    bool setPreprocess(const PreprocessConfig& config) override {
        options_.preprocess = config;
        return true;
    }

    /// Ngôn ngữ thực sự đang dùng (sau khi chọn theo gói đã cài).
    const std::string& language() const { return language_; }

private:
    /// Chạy một lượt nhận dạng với chế độ phân đoạn ``psm``.
    std::vector<OCRToken> runOnce(const cv::Mat& image, int psm);

    struct Impl;
    std::unique_ptr<Impl> impl_;
    TesseractOptions options_;
    std::string language_;
};

}  // namespace ctkm::ocr
