// Port của ``ctkm_extractor/extraction/field_parsers.py``.
//
// Nguyên tắc chung của mọi parser:
//   * nhận đầu vào là chuỗi bất kỳ (kể cả rỗng / nhiễu OCR),
//   * trả về giá trị đã parse HOẶC ``std::nullopt`` nếu không suy ra được,
//   * KHÔNG BAO GIỜ ném exception - lỗi được ghi log ở mức warning.
//
// Thêm parser mới chỉ cần đăng ký thêm một hàm vào registry rồi trỏ tên parser
// đó trong ``schema.json``; không phải sửa orchestrator.
#pragma once

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "util/Json.hpp"

namespace ctkm::extraction {

/// Giá trị một field sau khi parse.
using FieldValue = std::variant<double, long long, std::string, std::vector<std::string>>;

/// Tham số parser lấy từ ``parser_args`` trong schema.
class ParserArgs {
public:
    ParserArgs() = default;
    explicit ParserArgs(const json::Value& value) : value_(value) {}

    bool has(const std::string& key) const { return value_.contains(key); }
    std::string getString(const std::string& key, const std::string& fallback = "") const;
    bool getBool(const std::string& key, bool fallback) const;
    double getDouble(const std::string& key, double fallback) const;
    int getInt(const std::string& key, int fallback) const;
    std::vector<std::string> getStringArray(const std::string& key,
                                            const std::vector<std::string>& fallback = {}) const;
    std::optional<double> getOptionalDouble(const std::string& key) const;

private:
    json::Value value_;
};

using Parser = std::function<std::optional<FieldValue>(const std::string&, const ParserArgs&)>;

// ---------------------------------------------------------------------------
// Tiện ích chuẩn hoá chuỗi
// ---------------------------------------------------------------------------

/// Chuỗi đã bỏ dấu + ánh xạ ngược vị trí byte về chuỗi gốc.
///
/// Bản Python có ``fold_accents`` giữ nguyên ĐỘ DÀI (1 ký tự -> 1 ký tự) để dò
/// regex trên bản không dấu rồi cắt giá trị từ chuỗi gốc theo đúng vị trí. Với
/// UTF-8 byte-based của C++, độ dài byte thay đổi ("ố" 3 byte -> "o" 1 byte),
/// nên phải mang theo bảng ánh xạ ``offsets``: offsets[i] là vị trí byte trong
/// chuỗi GỐC ứng với byte thứ i của chuỗi đã bỏ dấu (có phần tử cuối = size()).
struct FoldedText {
    std::string folded;
    std::vector<std::size_t> offsets;

    /// Vị trí byte trong chuỗi gốc ứng với vị trí byte ``foldedIndex``.
    std::size_t originalOffset(std::size_t foldedIndex) const;
};

/// Bỏ dấu tiếng Việt, giữ ánh xạ vị trí về chuỗi gốc.
FoldedText foldAccents(const std::string& text);

/// Bỏ dấu tiếng Việt (chỉ lấy chuỗi kết quả).
std::string foldAccentsSimple(const std::string& text);

/// Chuẩn hoá nhãn để so khớp header: bỏ dấu, lowercase, gom khoảng trắng.
std::string normalizeLabel(const std::string& text);

/// Cắt bỏ khoảng trắng hai đầu.
std::string trim(const std::string& text);

/// Cắt bỏ các ký tự trong ``chars`` (danh sách ký tự UTF-8) ở hai đầu.
std::string stripCharacters(const std::string& text, const std::vector<std::string>& chars);

/// Bộ ký tự mặc định bị cắt hai đầu trong ``text_parser`` / ``list_parser``.
const std::vector<std::string>& defaultStripCharacters();

/// Sửa ký tự bị OCR nhầm dựa trên ngữ cảnh số ("1O0" -> "100", "150 sms" giữ nguyên).
std::string fixOcrDigits(const std::string& text);

/// Phần chuỗi nằm sau ``keyword`` (so khớp không dấu, không phân biệt hoa thường).
std::string sliceAfterKeyword(const std::string& text, const std::string& keyword);

/// True nếu ``keyword`` xuất hiện trong ``text`` (bỏ dấu, không phân biệt hoa thường).
bool containsKeyword(const std::string& text, const std::string& keyword);

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parse số tiền có thể lẫn lộn dấu ``,`` và ``.``.
std::optional<FieldValue> moneyParser(const std::string& raw, const ParserArgs& args);
/// Lấy số nguyên đầu tiên trong chuỗi.
std::optional<FieldValue> intParser(const std::string& raw, const ParserArgs& args);
/// Lấy dung lượng GB đầu tiên trong chuỗi.
std::optional<FieldValue> gbParser(const std::string& raw, const ParserArgs& args);
/// Trả chuỗi đã làm sạch; có thể lọc theo ``pattern``.
std::optional<FieldValue> textParser(const std::string& raw, const ParserArgs& args);
/// Trích từ khoá chu kỳ (``tháng``/``ngày``/``tuần``...).
std::optional<FieldValue> cycleParser(const std::string& raw, const ParserArgs& args);
/// Tách danh sách theo dấu phẩy thành mảng chuỗi.
std::optional<FieldValue> listParser(const std::string& raw, const ParserArgs& args);
/// Trả chuỗi gần như nguyên trạng (chỉ trim).
std::optional<FieldValue> rawParser(const std::string& raw, const ParserArgs& args);

/// Registry tên parser -> hàm.
const std::map<std::string, Parser>& parserRegistry();

/// Lấy parser theo tên; không có thì cảnh báo và dùng ``text_parser``.
Parser getParser(const std::string& name);

/// Gọi parser an toàn: mọi exception đều thành ``std::nullopt`` + log warning.
std::optional<FieldValue> parseValue(const std::string& parserName, const std::string& raw,
                                     const ParserArgs& args);

/// Danh sách tên parser đang đăng ký (tiện cho log/debug).
std::vector<std::string> availableParsers();

/// Chuyển giá trị field sang JSON để ghi ra output.
json::Value toJson(const std::optional<FieldValue>& value);

/// Biểu diễn ngắn gọn của giá trị (dùng cho ``--debug``).
std::string describe(const std::optional<FieldValue>& value);

}  // namespace ctkm::extraction
