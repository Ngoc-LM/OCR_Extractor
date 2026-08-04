#include "extraction/FieldParsers.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <regex>
#include <set>

#include "util/Log.hpp"
#include "util/Utf8.hpp"

namespace ctkm::extraction {
namespace {

constexpr const char* kLogger = "ctkm.extraction.field_parsers";

/// Ký tự phân cách hàng nghìn/thập phân.
const std::string kSeparators = ".,";

/// Ký tự vừa là chữ số bị OCR nhầm, vừa là ĐƯỜNG KẺ DỌC của bảng (port
/// ``OCR_BORDER_CONFUSIONS``).
///
/// Chỉ đổi thành chữ số khi nằm GIỮA hai chữ số. Nằm ở đầu/cuối cụm thì gần như
/// luôn là đường kẻ ô bị OCR đọc lẫn vào nội dung: ``"163,636.3636|"`` mà đổi
/// thành ``163636.36361`` là bịa thêm một chữ số vào một số vốn đã đọc đúng.
const std::string kBorderConfusions = "|!";

/// Ký tự thường bị OCR nhầm khi nằm cạnh chữ số (port ``OCR_DIGIT_CONFUSIONS``).
const std::map<char, char>& ocrDigitConfusions() {
    static const std::map<char, char> table = {
        {'O', '0'}, {'o', '0'}, {'Q', '0'}, {'D', '0'}, {'l', '1'}, {'I', '1'},
        {'|', '1'}, {'!', '1'}, {'S', '5'}, {'s', '5'}, {'B', '8'}, {'Z', '2'},
        {'z', '2'}, {'b', '6'}};
    return table;
}

/// Bảng bỏ dấu tiếng Việt (kèm các nguyên âm Latin-1 hay gặp).
const std::map<unsigned int, char>& accentTable() {
    static const std::map<unsigned int, char> table = [] {
        struct Entry {
            const char* accented;
            char base;
        };
        static const Entry entries[] = {
            {"àáảãạăằắẳẵặâầấẩẫậ", 'a'},
            {"ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ", 'A'},
            {"èéẻẽẹêềếểễệ", 'e'},
            {"ÈÉẺẼẸÊỀẾỂỄỆ", 'E'},
            {"ìíỉĩị", 'i'},
            {"ÌÍỈĨỊ", 'I'},
            {"òóỏõọôồốổỗộơờớởỡợ", 'o'},
            {"ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ", 'O'},
            {"ùúủũụưừứửữự", 'u'},
            {"ÙÚỦŨỤƯỪỨỬỮỰ", 'U'},
            {"ỳýỷỹỵ", 'y'},
            {"ỲÝỶỸỴ", 'Y'},
            {"đ", 'd'},
            {"Đ", 'D'},
            {"çñ", 'c'},
        };
        std::map<unsigned int, char> built;
        for (const auto& entry : entries) {
            const std::string source(entry.accented);
            std::size_t offset = 0;
            while (offset < source.size()) {
                const unsigned int codePoint = utf8::decode(source, offset);
                built[codePoint] = entry.base;
            }
        }
        // "ñ" bị gộp nhầm ở trên nếu để chung; đặt lại cho đúng.
        built[0x00F1] = 'n';
        built[0x00D1] = 'N';
        built[0x00E7] = 'c';
        built[0x00C7] = 'C';
        return built;
    }();
    return table;
}

char asciiLower(char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

char asciiUpper(char c) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
}

std::string toLowerAscii(const std::string& text) {
    std::string out = text;
    for (auto& c : out) {
        c = asciiLower(c);
    }
    return out;
}

bool isSpace(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v';
}

/// Regex một cụm số có thể kèm phân cách hàng nghìn/thập phân.
const std::regex& numberPattern() {
    static const std::regex pattern(R"([-+]?\d[\d\s.,]*\d|[-+]?\d)");
    return pattern;
}

/// Tách chuỗi số thành ``(phần nguyên, phần thập phân)``.
///
/// Heuristic (port nguyên văn bản Python):
///   * Có cả hai loại dấu -> dấu XUẤT HIỆN SAU CÙNG là dấu thập phân
///     ("163,636.3636" -> 163636.3636).
///   * Chỉ một loại, xuất hiện nhiều lần và mọi nhóm phía sau đều đúng 3 chữ số
///     -> toàn bộ là phân cách hàng nghìn ("150.534.213" -> 150534213).
///   * Chỉ một loại, xuất hiện một lần -> là hàng nghìn nếu nhóm sau đúng 3 chữ
///     số, ngược lại là dấu thập phân ("163,6363" -> 163.6363).
std::pair<std::string, std::string> splitNumberString(const std::string& cleaned) {
    std::vector<std::size_t> positions;
    for (std::size_t index = 0; index < cleaned.size(); ++index) {
        if (kSeparators.find(cleaned[index]) != std::string::npos) {
            positions.push_back(index);
        }
    }
    const auto removeSeparators = [](const std::string& text) {
        std::string out;
        for (const char c : text) {
            if (kSeparators.find(c) == std::string::npos) {
                out.push_back(c);
            }
        }
        return out;
    };
    if (positions.empty()) {
        return {cleaned, std::string()};
    }

    std::vector<std::string> groups;
    for (std::size_t order = 0; order < positions.size(); ++order) {
        const std::size_t end =
            order + 1 < positions.size() ? positions[order + 1] : cleaned.size();
        groups.push_back(cleaned.substr(positions[order] + 1, end - positions[order] - 1));
    }

    std::set<char> distinct;
    for (const auto position : positions) {
        distinct.insert(cleaned[position]);
    }

    bool hasDecimal = false;
    std::size_t decimalPosition = 0;
    if (distinct.size() > 1) {
        hasDecimal = true;
        decimalPosition = positions.back();
    } else if (positions.size() > 1) {
        bool allThree = true;
        for (const auto& group : groups) {
            if (group.size() != 3) {
                allThree = false;
                break;
            }
        }
        if (!allThree) {
            hasDecimal = true;
            decimalPosition = positions.back();
        }
    } else {
        if (groups.front().size() != 3) {
            hasDecimal = true;
            decimalPosition = positions.front();
        }
    }

    if (!hasDecimal) {
        return {removeSeparators(cleaned), std::string()};
    }
    std::string integerPart = removeSeparators(cleaned.substr(0, decimalPosition));
    std::string fractionPart;
    for (std::size_t index = decimalPosition + 1; index < cleaned.size(); ++index) {
        if (std::isdigit(static_cast<unsigned char>(cleaned[index])) != 0) {
            fractionPart.push_back(cleaned[index]);
        }
    }
    return {integerPart, fractionPart};
}

/// Chuyển chuỗi số đã làm sạch thành số nguyên hoặc số thực.
std::optional<FieldValue> toNumber(const std::string& raw) {
    std::string cleaned = raw;
    const int sign = (!cleaned.empty() && cleaned.front() == '-') ? -1 : 1;
    while (!cleaned.empty() && (cleaned.front() == '+' || cleaned.front() == '-')) {
        cleaned.erase(cleaned.begin());
    }
    cleaned = trim(cleaned);
    // strip('.,') hai đầu như bản Python.
    while (!cleaned.empty() && kSeparators.find(cleaned.front()) != std::string::npos) {
        cleaned.erase(cleaned.begin());
    }
    while (!cleaned.empty() && kSeparators.find(cleaned.back()) != std::string::npos) {
        cleaned.pop_back();
    }
    if (cleaned.empty()) {
        return std::nullopt;
    }
    const bool hasDigit = std::any_of(cleaned.begin(), cleaned.end(), [](char c) {
        return std::isdigit(static_cast<unsigned char>(c)) != 0;
    });
    if (!hasDigit) {
        return std::nullopt;
    }

    auto [integerPart, fractionPart] = splitNumberString(cleaned);
    if (integerPart.empty()) {
        integerPart = "0";
    }
    const bool fractionIsZero =
        fractionPart.empty() ||
        std::all_of(fractionPart.begin(), fractionPart.end(), [](char c) { return c == '0'; });
    if (fractionIsZero) {
        try {
            const long long value = std::stoll(integerPart);
            return FieldValue{static_cast<long long>(sign) * value};
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }
    try {
        const double value = std::stod(integerPart + "." + fractionPart);
        return FieldValue{static_cast<double>(sign) * value};
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

/// Lấy cụm số thứ ``occurrence`` trong chuỗi (đã bỏ khoảng trắng bên trong).
std::optional<std::string> findNumberString(const std::string& text, int occurrence) {
    const std::regex& pattern = numberPattern();
    auto begin = std::sregex_iterator(text.begin(), text.end(), pattern);
    const auto end = std::sregex_iterator();
    int index = 0;
    for (auto it = begin; it != end; ++it, ++index) {
        if (index == occurrence) {
            std::string match = it->str();
            std::string cleaned;
            for (const char c : match) {
                if (!isSpace(c)) {
                    cleaned.push_back(c);
                }
            }
            return cleaned;
        }
    }
    return std::nullopt;
}

/// Chuyển ``FieldValue`` số về double để so ngưỡng min/max.
std::optional<double> numericValue(const FieldValue& value) {
    if (std::holds_alternative<long long>(value)) {
        return static_cast<double>(std::get<long long>(value));
    }
    if (std::holds_alternative<double>(value)) {
        return std::get<double>(value);
    }
    return std::nullopt;
}

std::string formatDouble(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.10g", value);
    return buffer;
}

}  // namespace

// ---------------------------------------------------------------------------
// ParserArgs
// ---------------------------------------------------------------------------
std::string ParserArgs::getString(const std::string& key, const std::string& fallback) const {
    const json::Value* found = value_.find(key);
    if (found == nullptr || !found->isString()) {
        return fallback;
    }
    return found->asString();
}

bool ParserArgs::getBool(const std::string& key, bool fallback) const {
    const json::Value* found = value_.find(key);
    if (found == nullptr || !found->isBool()) {
        return fallback;
    }
    return found->asBool();
}

double ParserArgs::getDouble(const std::string& key, double fallback) const {
    const json::Value* found = value_.find(key);
    if (found == nullptr || !found->isNumber()) {
        return fallback;
    }
    return found->asDouble();
}

int ParserArgs::getInt(const std::string& key, int fallback) const {
    const json::Value* found = value_.find(key);
    if (found == nullptr || !found->isNumber()) {
        return fallback;
    }
    return static_cast<int>(found->asInteger(fallback));
}

std::vector<std::string> ParserArgs::getStringArray(
    const std::string& key, const std::vector<std::string>& fallback) const {
    const json::Value* found = value_.find(key);
    if (found == nullptr || !found->isArray()) {
        return fallback;
    }
    const auto items = found->asStringArray();
    return items.empty() ? fallback : items;
}

std::optional<double> ParserArgs::getOptionalDouble(const std::string& key) const {
    const json::Value* found = value_.find(key);
    if (found == nullptr || !found->isNumber()) {
        return std::nullopt;
    }
    return found->asDouble();
}

// ---------------------------------------------------------------------------
// Tiện ích chuỗi
// ---------------------------------------------------------------------------
std::size_t FoldedText::originalOffset(std::size_t foldedIndex) const {
    if (offsets.empty()) {
        return 0;
    }
    if (foldedIndex >= offsets.size()) {
        return offsets.back();
    }
    return offsets[foldedIndex];
}

FoldedText foldAccents(const std::string& text) {
    FoldedText result;
    const auto& table = accentTable();
    std::size_t offset = 0;
    while (offset < text.size()) {
        const std::size_t start = offset;
        const unsigned int codePoint = utf8::decode(text, offset);
        const auto found = table.find(codePoint);
        if (found != table.end()) {
            result.offsets.push_back(start);
            result.folded.push_back(found->second);
        } else {
            for (std::size_t index = start; index < offset; ++index) {
                result.offsets.push_back(index);
                result.folded.push_back(text[index]);
            }
        }
    }
    result.offsets.push_back(text.size());
    return result;
}

std::string foldAccentsSimple(const std::string& text) { return foldAccents(text).folded; }

std::string trim(const std::string& text) {
    std::size_t begin = 0;
    while (begin < text.size() && isSpace(text[begin])) {
        ++begin;
    }
    std::size_t end = text.size();
    while (end > begin && isSpace(text[end - 1])) {
        --end;
    }
    return text.substr(begin, end - begin);
}

std::string normalizeLabel(const std::string& text) {
    const std::string folded = toLowerAscii(foldAccentsSimple(text));
    std::string collapsed;
    bool pendingSpace = false;
    for (const char c : folded) {
        const bool keep = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '+';
        if (keep) {
            if (pendingSpace && !collapsed.empty()) {
                collapsed.push_back(' ');
            }
            pendingSpace = false;
            collapsed.push_back(c);
        } else {
            pendingSpace = true;
        }
    }
    return collapsed;
}

const std::vector<std::string>& defaultStripCharacters() {
    static const std::vector<std::string> chars = {" ",  "\t", "\r", "\n", ":",
                                                   ";",  ".",  "-",  "–",  "—",
                                                   "|"};
    return chars;
}

std::string stripCharacters(const std::string& text, const std::vector<std::string>& chars) {
    std::string out = text;
    bool changed = true;
    while (changed && !out.empty()) {
        changed = false;
        for (const auto& candidate : chars) {
            if (candidate.empty()) {
                continue;
            }
            if (out.size() >= candidate.size() &&
                out.compare(0, candidate.size(), candidate) == 0) {
                out.erase(0, candidate.size());
                changed = true;
                break;
            }
        }
        for (const auto& candidate : chars) {
            if (candidate.empty() || out.size() < candidate.size()) {
                continue;
            }
            if (out.compare(out.size() - candidate.size(), candidate.size(), candidate) == 0) {
                out.erase(out.size() - candidate.size());
                changed = true;
                break;
            }
        }
    }
    return out;
}

std::string fixOcrDigits(const std::string& text) {
    if (text.empty()) {
        return std::string();
    }
    const auto& confusions = ocrDigitConfusions();

    // chunk[index] có được phép đổi thành chữ số không?
    const auto convertible = [&confusions](const std::string& chunk, std::size_t index) {
        if (confusions.find(chunk[index]) == confusions.end()) {
            return false;
        }
        const bool previousDigit =
            index > 0 && std::isdigit(static_cast<unsigned char>(chunk[index - 1])) != 0;
        const bool nextDigit = index + 1 < chunk.size() &&
                               std::isdigit(static_cast<unsigned char>(chunk[index + 1])) != 0;
        if (kBorderConfusions.find(chunk[index]) != std::string::npos) {
            // Phải nằm TRONG LÒNG cụm số (hai bên đều là chữ số hoặc dấu phân
            // cách) mới là chữ số; ở rìa cụm thì là đường kẻ ô.
            const bool previousInside =
                previousDigit || (index > 0 && kSeparators.find(chunk[index - 1]) !=
                                                   std::string::npos);
            const bool nextInside =
                nextDigit || (index + 1 < chunk.size() &&
                              kSeparators.find(chunk[index + 1]) != std::string::npos);
            return previousInside && nextInside && (previousDigit || nextDigit);
        }
        return previousDigit || nextDigit;
    };

    const auto fixChunk = [&confusions, &convertible](const std::string& chunk) {
        if (chunk.empty()) {
            return chunk;
        }
        // Mức 1: cụm chỉ gồm chữ số, dấu phân cách và ký tự dễ nhầm -> sửa hết,
        // trừ đường kẻ ô ở rìa cụm.
        const bool hasDigit = std::any_of(chunk.begin(), chunk.end(), [](char c) {
            return std::isdigit(static_cast<unsigned char>(c)) != 0;
        });
        const bool onlyNumericish = std::all_of(chunk.begin(), chunk.end(), [&](char c) {
            return std::isdigit(static_cast<unsigned char>(c)) != 0 ||
                   kSeparators.find(c) != std::string::npos ||
                   confusions.find(c) != confusions.end();
        });
        std::string out = chunk;
        if (hasDigit && onlyNumericish) {
            for (std::size_t index = 0; index < chunk.size(); ++index) {
                if (kBorderConfusions.find(chunk[index]) != std::string::npos &&
                    !convertible(chunk, index)) {
                    continue;
                }
                const auto found = confusions.find(chunk[index]);
                if (found != confusions.end()) {
                    out[index] = found->second;
                }
            }
            return out;
        }
        // Mức 2: chỉ sửa ký tự LIỀN KỀ một chữ số.
        for (std::size_t index = 0; index < chunk.size(); ++index) {
            if (convertible(chunk, index)) {
                out[index] = confusions.find(chunk[index])->second;
            }
        }
        return out;
    };

    std::string result;
    std::string chunk;
    for (const char c : text) {
        if (isSpace(c)) {
            if (!chunk.empty()) {
                result += fixChunk(chunk);
                chunk.clear();
            }
            result.push_back(c);
        } else {
            chunk.push_back(c);
        }
    }
    if (!chunk.empty()) {
        result += fixChunk(chunk);
    }
    return result;
}

std::string sliceAfterKeyword(const std::string& text, const std::string& keyword) {
    if (keyword.empty()) {
        return text;
    }
    const FoldedText folded = foldAccents(text);
    const std::string haystack = toLowerAscii(folded.folded);
    const std::string needle = toLowerAscii(foldAccentsSimple(keyword));
    const std::size_t position = haystack.find(needle);
    if (position == std::string::npos) {
        return text;
    }
    const std::size_t originalEnd = folded.originalOffset(position + needle.size());
    return text.substr(originalEnd);
}

bool containsKeyword(const std::string& text, const std::string& keyword) {
    const std::string haystack = toLowerAscii(foldAccentsSimple(text));
    const std::string needle = toLowerAscii(foldAccentsSimple(keyword));
    return haystack.find(needle) != std::string::npos;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------
std::optional<FieldValue> moneyParser(const std::string& raw, const ParserArgs& args) {
    const std::string text = trim(raw);
    if (text.empty()) {
        return std::nullopt;
    }
    std::string scoped = sliceAfterKeyword(text, args.getString("keyword"));
    if (args.getBool("fix_ocr", true)) {
        scoped = fixOcrDigits(scoped);
    }
    const auto numberString = findNumberString(scoped, args.getInt("occurrence", 0));
    if (!numberString.has_value()) {
        log::debug(kLogger, "money_parser không tìm thấy số trong '" + text + "'");
        return std::nullopt;
    }
    const auto value = toNumber(*numberString);
    if (!value.has_value()) {
        return std::nullopt;
    }
    const auto numeric = numericValue(*value);
    if (numeric.has_value()) {
        const auto minValue = args.getOptionalDouble("min_value");
        if (minValue.has_value() && *numeric < *minValue) {
            log::warn(kLogger, "money_parser: giá trị " + formatDouble(*numeric) +
                                   " nhỏ hơn ngưỡng " + formatDouble(*minValue));
            return std::nullopt;
        }
        const auto maxValue = args.getOptionalDouble("max_value");
        if (maxValue.has_value() && *numeric > *maxValue) {
            log::warn(kLogger, "money_parser: giá trị " + formatDouble(*numeric) +
                                   " lớn hơn ngưỡng " + formatDouble(*maxValue));
            return std::nullopt;
        }
    }
    return value;
}

std::optional<FieldValue> intParser(const std::string& raw, const ParserArgs& args) {
    // Bỏ min/max khỏi lượt gọi money_parser để kiểm tra ngưỡng đúng trên số nguyên
    // đã làm tròn, giống bản Python (money_parser được gọi không kèm ngưỡng).
    json::Object forwarded;
    if (args.has("keyword")) {
        forwarded.emplace_back("keyword", json::Value(args.getString("keyword")));
    }
    forwarded.emplace_back("occurrence", json::Value(args.getInt("occurrence", 0)));
    forwarded.emplace_back("fix_ocr", json::Value(args.getBool("fix_ocr", true)));
    const ParserArgs inner{json::Value(forwarded)};

    const auto value = moneyParser(raw, inner);
    if (!value.has_value()) {
        return std::nullopt;
    }
    const auto numeric = numericValue(*value);
    if (!numeric.has_value()) {
        log::warn(kLogger, "int_parser không ép được giá trị về số nguyên");
        return std::nullopt;
    }
    const long long rounded = static_cast<long long>(std::llround(*numeric));

    const auto minValue = args.getOptionalDouble("min_value");
    if (minValue.has_value() && static_cast<double>(rounded) < *minValue) {
        log::warn(kLogger, "int_parser: giá trị " + std::to_string(rounded) +
                               " nhỏ hơn ngưỡng " + formatDouble(*minValue));
        return std::nullopt;
    }
    const auto maxValue = args.getOptionalDouble("max_value");
    if (maxValue.has_value() && static_cast<double>(rounded) > *maxValue) {
        log::warn(kLogger, "int_parser: giá trị " + std::to_string(rounded) +
                               " lớn hơn ngưỡng " + formatDouble(*maxValue));
        return std::nullopt;
    }
    return FieldValue{rounded};
}

std::optional<FieldValue> gbParser(const std::string& raw, const ParserArgs& args) {
    const std::string text = trim(raw);
    if (text.empty()) {
        return std::nullopt;
    }
    const std::string keyword = args.getString("keyword");
    if (!keyword.empty() && args.getBool("require_keyword", false) &&
        !containsKeyword(text, keyword)) {
        log::debug(kLogger, "gb_parser: không thấy keyword '" + keyword + "'");
        return std::nullopt;
    }
    std::string scoped = sliceAfterKeyword(text, keyword);
    if (args.getBool("fix_ocr", true)) {
        scoped = fixOcrDigits(scoped);
    }

    const std::string unitPattern = args.getString("unit_pattern", R"(g\s?b)");
    try {
        const std::regex pattern(R"(([-+]?\d[\d\s.,]*\d|\d)\s*)" + unitPattern,
                                 std::regex::icase);
        std::smatch match;
        if (std::regex_search(scoped, match, pattern)) {
            std::string number = match[1].str();
            std::string cleaned;
            for (const char c : number) {
                if (!isSpace(c)) {
                    cleaned.push_back(c);
                }
            }
            return toNumber(cleaned);
        }
    } catch (const std::regex_error& error) {
        log::warn(kLogger, std::string("gb_parser: unit_pattern không hợp lệ: ") + error.what());
    }

    if (args.getBool("fallback_to_number", true)) {
        const auto numberString = findNumberString(scoped, 0);
        if (numberString.has_value()) {
            return toNumber(*numberString);
        }
    }
    log::debug(kLogger, "gb_parser không tìm thấy dung lượng trong '" + text + "'");
    return std::nullopt;
}

std::optional<FieldValue> textParser(const std::string& raw, const ParserArgs& args) {
    const std::string source = trim(raw);
    if (source.empty()) {
        return std::nullopt;
    }
    std::vector<std::string> stripChars = defaultStripCharacters();
    if (args.has("strip_chars")) {
        stripChars = utf8::toCharacters(args.getString("strip_chars"));
    }

    std::string text = sliceAfterKeyword(source, args.getString("keyword"));
    text = trim(stripCharacters(trim(text), stripChars));

    const std::string pattern = args.getString("pattern");
    if (!pattern.empty()) {
        try {
            const std::regex expression(pattern, std::regex::icase);
            std::smatch match;
            if (!std::regex_search(text, match, expression)) {
                log::debug(kLogger, "text_parser: '" + text + "' không khớp pattern");
                return std::nullopt;
            }
            text = match.size() > 1 ? match[1].str() : match[0].str();
        } catch (const std::regex_error& error) {
            log::warn(kLogger, "text_parser: pattern '" + pattern + "' không hợp lệ (" +
                                   error.what() + ")");
            return std::nullopt;
        }
    }

    text = trim(stripCharacters(trim(text), stripChars));
    if (text.empty()) {
        return std::nullopt;
    }
    const int maxLength = args.getInt("max_length", -1);
    if (maxLength >= 0 && utf8::length(text) > static_cast<std::size_t>(maxLength)) {
        text = trim(utf8::substr(text, 0, static_cast<std::size_t>(maxLength)));
    }
    if (args.getBool("upper", false)) {
        for (auto& c : text) {
            c = asciiUpper(c);
        }
    } else if (args.getBool("lower", false)) {
        for (auto& c : text) {
            c = asciiLower(c);
        }
    }
    if (text.empty()) {
        return std::nullopt;
    }
    return FieldValue{text};
}

std::optional<FieldValue> cycleParser(const std::string& raw, const ParserArgs& args) {
    const std::string text = trim(raw);
    if (text.empty()) {
        return std::nullopt;
    }
    const std::vector<std::string> candidates =
        args.getStringArray("keywords", {"tháng", "ngày", "tuần", "quý", "năm"});
    const std::vector<std::string> prefixes = args.getStringArray(
        "prefer_after", {"chu kỳ", "chu ky", "cycle", "gia hạn", "gia han"});

    std::string scoped = text;
    const FoldedText foldedFull = foldAccents(text);
    const std::string haystack = toLowerAscii(foldedFull.folded);
    for (const auto& prefix : prefixes) {
        const std::string needle = toLowerAscii(foldAccentsSimple(prefix));
        const std::size_t position = haystack.find(needle);
        if (position != std::string::npos) {
            scoped = text.substr(foldedFull.originalOffset(position + needle.size()));
            break;
        }
    }

    const bool lower = args.getBool("lower", true);
    for (const std::string& source : {scoped, text}) {
        const std::string folded = toLowerAscii(foldAccentsSimple(source));
        std::size_t bestPosition = std::string::npos;
        std::string bestKeyword;
        for (const auto& candidate : candidates) {
            const std::string needle = toLowerAscii(foldAccentsSimple(candidate));
            const std::size_t position = folded.find(needle);
            if (position != std::string::npos &&
                (bestPosition == std::string::npos || position < bestPosition)) {
                bestPosition = position;
                bestKeyword = candidate;
            }
        }
        if (!bestKeyword.empty()) {
            return FieldValue{lower ? toLowerAscii(bestKeyword) : bestKeyword};
        }
    }
    log::debug(kLogger, "cycle_parser không tìm thấy từ khoá chu kỳ trong '" + text + "'");
    return std::nullopt;
}

std::optional<FieldValue> listParser(const std::string& raw, const ParserArgs& args) {
    const std::string source = trim(raw);
    if (source.empty()) {
        return std::nullopt;
    }
    const std::string text = sliceAfterKeyword(source, args.getString("keyword"));
    const std::vector<std::string> separators =
        args.getStringArray("separators", {",", ";", "\n", "/", "|"});
    std::vector<std::string> stripChars = {" ", "\t", "\r", "\n", ".", ";",
                                           ":", "|",  "-",  "–",  "—"};
    if (args.has("strip_chars")) {
        stripChars = utf8::toCharacters(args.getString("strip_chars"));
    }
    const std::size_t minItemLength =
        static_cast<std::size_t>(std::max(1, args.getInt("min_item_length", 1)));
    const int maxItems = args.getInt("max_items", -1);
    const bool unique = args.getBool("unique", true);

    // Tách theo bất kỳ separator nào (tương đương re.split với các separator escape).
    std::vector<std::string> parts;
    std::string current;
    std::size_t index = 0;
    while (index < text.size()) {
        bool matched = false;
        for (const auto& separator : separators) {
            if (separator.empty() || index + separator.size() > text.size()) {
                continue;
            }
            if (text.compare(index, separator.size(), separator) == 0) {
                parts.push_back(current);
                current.clear();
                index += separator.size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            current.push_back(text[index]);
            ++index;
        }
    }
    parts.push_back(current);

    std::vector<std::string> items;
    for (const auto& part : parts) {
        const std::string cleaned = trim(stripCharacters(trim(part), stripChars));
        if (utf8::length(cleaned) < minItemLength) {
            continue;
        }
        if (unique && std::find(items.begin(), items.end(), cleaned) != items.end()) {
            continue;
        }
        items.push_back(cleaned);
        if (maxItems >= 0 && static_cast<int>(items.size()) >= maxItems) {
            break;
        }
    }
    if (items.empty()) {
        log::debug(kLogger, "list_parser không tách được phần tử nào từ '" + text + "'");
        return std::nullopt;
    }
    return FieldValue{items};
}

std::optional<FieldValue> rawParser(const std::string& raw, const ParserArgs& args) {
    (void)args;
    const std::string text = trim(raw);
    if (text.empty()) {
        return std::nullopt;
    }
    return FieldValue{text};
}

const std::map<std::string, Parser>& parserRegistry() {
    static const std::map<std::string, Parser> registry = {
        {"money_parser", moneyParser}, {"int_parser", intParser},
        {"gb_parser", gbParser},       {"text_parser", textParser},
        {"cycle_parser", cycleParser}, {"list_parser", listParser},
        {"raw_parser", rawParser}};
    return registry;
}

Parser getParser(const std::string& name) {
    const auto& registry = parserRegistry();
    const auto found = registry.find(name);
    if (found == registry.end()) {
        log::warn(kLogger, "Không tìm thấy parser '" + name + "', dùng text_parser thay thế");
        return registry.at("text_parser");
    }
    return found->second;
}

std::optional<FieldValue> parseValue(const std::string& parserName, const std::string& raw,
                                     const ParserArgs& args) {
    try {
        return getParser(parserName)(raw, args);
    } catch (const std::exception& error) {
        log::warn(kLogger, "Parser '" + parserName + "' lỗi với giá trị '" + raw +
                               "': " + error.what());
        return std::nullopt;
    }
}

std::vector<std::string> availableParsers() {
    std::vector<std::string> names;
    for (const auto& entry : parserRegistry()) {
        names.push_back(entry.first);
    }
    return names;
}

json::Value toJson(const std::optional<FieldValue>& value) {
    if (!value.has_value()) {
        return json::Value(nullptr);
    }
    if (std::holds_alternative<long long>(*value)) {
        return json::Value(std::get<long long>(*value));
    }
    if (std::holds_alternative<double>(*value)) {
        return json::Value(std::get<double>(*value));
    }
    if (std::holds_alternative<std::string>(*value)) {
        return json::Value(std::get<std::string>(*value));
    }
    json::Array items;
    for (const auto& item : std::get<std::vector<std::string>>(*value)) {
        items.push_back(json::Value(item));
    }
    return json::Value(std::move(items));
}

std::string describe(const std::optional<FieldValue>& value) {
    if (!value.has_value()) {
        return "None";
    }
    if (std::holds_alternative<long long>(*value)) {
        return std::to_string(std::get<long long>(*value));
    }
    if (std::holds_alternative<double>(*value)) {
        return formatDouble(std::get<double>(*value));
    }
    if (std::holds_alternative<std::string>(*value)) {
        return "'" + std::get<std::string>(*value) + "'";
    }
    std::string out = "[";
    const auto& items = std::get<std::vector<std::string>>(*value);
    for (std::size_t index = 0; index < items.size(); ++index) {
        if (index > 0) {
            out += ", ";
        }
        out += "'" + items[index] + "'";
    }
    out += "]";
    return out;
}

}  // namespace ctkm::extraction
