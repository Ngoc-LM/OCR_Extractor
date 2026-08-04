// Port của ``ctkm_extractor/extraction/extractor.py``.
//
// Luồng xử lý cho mỗi field, dừng ở bước đầu tiên cho kết quả:
//   1. Tìm nhãn (alias) trong BẢNG đã dựng, lấy ô giá trị bên phải/bên dưới.
//   2. Tìm nhãn theo TỪNG DÒNG của raw text.
//   3. Dò REGEX fallback khai báo trong schema trên toàn bộ raw text.
//   4. Không tìm được -> ghi log warning và trả null (không ném exception).
//
// Sau khi resolve xong toàn bộ field còn một bước hậu kiểm chống 2 field cùng
// lấy giá trị từ một ô bảng (xem :func:`CTKMExtractor::resolveCellCollisions`).
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "extraction/FieldParsers.hpp"
#include "extraction/Schema.hpp"
#include "ocr/OCRProvider.hpp"
#include "ocr/ProviderFactory.hpp"
#include "table/Reconstruct.hpp"
#include "table/TableBuilder.hpp"
#include "util/Json.hpp"

namespace ctkm::extraction {

extern const char* const kSourceTable;
extern const char* const kSourceTextLine;
extern const char* const kSourceRegex;
extern const char* const kSourceMissing;

/// Ký tự phân tách nhãn với giá trị khi cả hai nằm trong cùng một ô.
const std::vector<std::string>& labelValueSeparators();

/// Kết quả trích xuất của một field.
struct FieldResult {
    std::string name;
    std::optional<FieldValue> value;
    std::optional<std::string> rawValue;
    std::string source = kSourceMissing;
    double score = 0.0;
    std::optional<std::string> matchedAlias;
    /// Toạ độ ô bảng đã cung cấp giá trị (chỉ có khi source == kSourceTable),
    /// dùng để phát hiện khi 2 field vô tình cùng lấy chung một ô.
    std::optional<std::pair<int, int>> cellCoords;
    /// Trang (1-based) đã cho ra giá trị này; rỗng khi đầu vào chỉ có một ảnh.
    std::optional<int> sourcePage;

    bool found() const { return value.has_value(); }
};

/// Kết quả trích xuất toàn bộ ảnh/khối text.
struct ExtractionResult {
    /// Giữ đúng thứ tự field khai báo trong schema.
    std::vector<FieldResult> fields;
    std::string rawText;
    std::optional<table::Table> table;
    std::string tableStrategy = "raw_text";
    std::string ocrProvider = "unknown";
    std::vector<std::string> warnings;
    /// Trang được chọn làm trang chính; rỗng khi đầu vào chỉ có một ảnh.
    std::optional<int> page;
    /// Số trang/ảnh đã OCR (0 khi chỉ xử lý một ảnh).
    int pagesProcessed = 0;

    /// Tra field theo tên; nullptr nếu không có.
    const FieldResult* field(const std::string& name) const;
    /// JSON output: đúng thứ tự field trong schema, thiếu thì null.
    json::Value toJson() const;
    /// Chuỗi JSON UTF-8 (không escape tiếng Việt).
    std::string toJsonString(int indent = 2) const;
    /// Báo cáo chi tiết dùng cho ``--debug`` (format giống bản Python).
    std::string debugReport() const;
};

/// Độ tương đồng 0..1 giữa hai nhãn đã chuẩn hoá.
///
/// Port thuật toán ``difflib.SequenceMatcher.ratio`` của Python (khớp khối chung
/// dài nhất theo đệ quy) để hai bản cho cùng điểm số - ngưỡng 0.82 và việc so
/// score trong collision guard phụ thuộc trực tiếp vào giá trị này.
double similarity(const std::string& left, const std::string& right);

/// Điểm khớp tốt nhất giữa nội dung một ô và danh sách alias của field.
struct AliasMatch {
    double score = 0.0;
    std::optional<std::string> alias;
};
AliasMatch matchAlias(const std::string& text, const std::vector<std::string>& aliases,
                      double minSimilarity);

/// Phần còn lại của ô sau khi bỏ nhãn ("Chu kỳ: tháng" -> "tháng").
std::string remainderAfterAlias(const std::string& text, const std::string& alias);

/// Alias ngắn hơn chừng này ký tự không được dùng làm mốc cắt giá trị: alias
/// quá ngắn ("SMS", "Data") dễ khớp nhầm vào giữa một cụm từ bình thường và
/// cắt cụt giá trị hợp lệ.
inline constexpr std::size_t kMinBoundaryAliasLength = 5;

/// Cắt giá trị tại vị trí nhãn của FIELD KHÁC bắt đầu.
///
/// Cần thiết vì OCR trả về nguyên một HÀNG của bảng thành MỘT dòng text: phần
/// còn lại của dòng sau nhãn chứa luôn mọi cột phía sau nó. Không cắt thì
/// parser vớ phải số của cột khác - đo trên biểu mẫu BM.12 thật, ô giá trị của
/// "Cước TB" bị OCR bỏ sót nên field monthlyFee nhận cả phần đuôi
/// "Thoại ngoại mạng ... (TK533) ..." và trả ra 533, là số của một cột hoàn
/// toàn khác.
///
/// Alias của chính field đang xét không cắt (nhãn có thể lặp lại trong chính
/// phần giá trị).
std::string cutAtNextLabel(const std::string& value,
                           const std::vector<std::string>& ownAliases,
                           const std::vector<std::string>& otherAliases);

/// Điểm của một trang: (số field trích được, điểm khớp trung bình).
///
/// Trang không chứa bảng CTKM ra (0, 0.0) nên không bao giờ được chọn làm trang
/// chính khi tồn tại trang khác có dữ liệu.
std::pair<int, double> pageExtractionScore(const ExtractionResult& result);

/// Gộp kết quả của nhiều trang/ảnh thành một kết quả duy nhất.
///
/// Bảng CTKM thường chỉ nằm ở MỘT trang trong cả tập hồ sơ:
///   1. Trang chính = trang điểm cao nhất (hoà thì lấy trang số nhỏ hơn).
///   2. Lấy toàn bộ giá trị của trang chính làm nền.
///   3. Field nào VẪN THIẾU mới lấy bù từ các trang còn lại theo điểm giảm dần.
///
/// Bước 3 cố tình không ghi đè giá trị đã có: trang chính là trang thật sự chứa
/// bảng, còn trang khác dễ có nhãn trùng tên trong phần văn bản thường.
ExtractionResult mergePageResults(
    const std::vector<std::pair<int, ExtractionResult>>& pages);

/// Tuỳ chọn của orchestrator (đặt ở namespace scope để dùng được làm default arg).
struct ExtractorOptions {
    std::string engine = "paddle_vietocr";
    bool useMorphology = true;
    bool usePpStructure = true;
    bool strictEngine = false;
    /// true/false = ép bật/tắt nhị phân hoá; rỗng = TỰ CHỌN (chạy cả hai
    /// rồi giữ kết quả trích được nhiều field hơn).
    std::optional<bool> binarize;
    /// Tham số dựng provider (đường dẫn model, tiền xử lý, ...).
    ocr::ProviderOptions provider;
};

/// Điều phối toàn bộ pipeline trích xuất CTKM.
class CTKMExtractor {
public:
    using Options = ExtractorOptions;

    /// Khởi tạo với schema đã nạp sẵn.
    explicit CTKMExtractor(Schema schema, Options options = Options());
    /// Khởi tạo từ đường dẫn schema (rỗng = schema mặc định).
    explicit CTKMExtractor(const std::string& schemaPath = "", Options options = Options());

    /// Inject provider dựng sẵn (test hoặc mock).
    void setProvider(std::shared_ptr<ocr::OCRProvider> provider) {
        provider_ = std::move(provider);
    }

    const Schema& schema() const { return schema_; }

    /// Chạy toàn bộ pipeline trên một ảnh.
    ExtractionResult extractFromImage(const std::string& imagePath);
    /// Một lượt OCR với một cấu hình tiền xử lý cụ thể.
    ExtractionResult extractImageOnce(const std::string& imagePath,
                                      const std::optional<ocr::PreprocessConfig>& config);
    /// Chạy trên NHIỀU ảnh (VD từng trang PDF đã render sẵn) rồi gộp kết quả.
    ExtractionResult extractFromImages(const std::vector<std::string>& imagePaths);
    /// Dựng bảng từ kết quả OCR rồi trích xuất field.
    ExtractionResult extractFromOcr(const ocr::OCRResult& ocrResult);
    /// Trích xuất từ một khối raw text (fixture test, hoặc OCR đã có sẵn).
    ExtractionResult extractFromText(const std::string& text);
    /// Trích xuất từ một bảng đã dựng sẵn (test tầng extraction độc lập).
    ExtractionResult extractFromTable(const table::Table& tableValue,
                                      const std::string& rawText = "");

private:
    /// Ứng viên giá trị cho một field, kèm nguồn và toạ độ ô (nếu từ bảng).
    struct Candidate {
        std::string source;
        std::string rawValue;
        double score = 0.0;
        std::optional<std::string> alias;
        std::optional<std::pair<int, int>> cellCoords;
    };

    ExtractionResult extract(const table::TableBuildResult& buildResult,
                             const std::string& providerName,
                             const std::vector<std::string>& extraWarnings);

    FieldResult resolveField(const FieldSpec& spec,
                             const table::TableBuildResult& buildResult) const;

    /// Chiến lược 1: tra trong bảng, trả danh sách ứng viên theo thứ tự ưu tiên.
    std::vector<Candidate> findInTable(const FieldSpec& spec,
                                       const table::Table& tableValue) const;
    /// Chiến lược 2: tra theo từng dòng raw text.
    std::optional<Candidate> findInTextLines(const FieldSpec& spec,
                                             const std::string& rawText) const;
    /// Chiến lược 3: regex fallback trên raw text blob.
    std::optional<std::string> findByRegex(const FieldSpec& spec,
                                           const std::string& rawText) const;

    /// True nếu ô trông giống NHÃN của một field nào đó chứ không phải giá trị.
    bool looksLikeLabel(const std::string& text) const;

    /// Ép/kiểm tra giá trị theo kiểu khai báo trong schema.
    std::optional<FieldValue> coerceType(const FieldSpec& spec,
                                         const std::optional<FieldValue>& value) const;

    /// Phát hiện 2 field khác nhau vô tình lấy giá trị từ cùng một ô bảng.
    std::vector<std::string> resolveCellCollisions(std::vector<FieldResult>& results) const;

    ocr::OCRProvider& provider();

    Schema schema_;
    Options options_;
    std::shared_ptr<ocr::OCRProvider> provider_;
};

}  // namespace ctkm::extraction
