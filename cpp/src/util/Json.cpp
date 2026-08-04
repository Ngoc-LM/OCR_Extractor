#include "util/Json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "util/Utf8.hpp"

namespace ctkm::json {
namespace {

/// Parser đệ quy xuống, đủ dùng cho schema và output của chương trình.
class Parser {
public:
    explicit Parser(const std::string& text) : text_(text) {}

    Value parse() {
        skipWhitespace();
        Value value = parseValue();
        skipWhitespace();
        if (position_ != text_.size()) {
            fail("thừa dữ liệu sau giá trị JSON");
        }
        return value;
    }

private:
    [[noreturn]] void fail(const std::string& message) const {
        throw JsonError("JSON lỗi tại byte " + std::to_string(position_) + ": " + message);
    }

    char peek() const {
        if (position_ >= text_.size()) {
            return '\0';
        }
        return text_[position_];
    }

    void skipWhitespace() {
        while (position_ < text_.size()) {
            const char c = text_[position_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++position_;
            } else if (c == '/' && position_ + 1 < text_.size() && text_[position_ + 1] == '/') {
                // Cho phép comment dòng "//" để schema.json chú thích được như YAML.
                while (position_ < text_.size() && text_[position_] != '\n') {
                    ++position_;
                }
            } else {
                break;
            }
        }
    }

    Value parseValue() {
        switch (peek()) {
            case '{':
                return parseObject();
            case '[':
                return parseArray();
            case '"':
                return Value(parseString());
            case 't':
                expect("true");
                return Value(true);
            case 'f':
                expect("false");
                return Value(false);
            case 'n':
                expect("null");
                return Value(nullptr);
            default:
                return parseNumber();
        }
    }

    void expect(const std::string& literal) {
        if (text_.compare(position_, literal.size(), literal) != 0) {
            fail("mong đợi '" + literal + "'");
        }
        position_ += literal.size();
    }

    Value parseObject() {
        Object members;
        ++position_;  // '{'
        skipWhitespace();
        if (peek() == '}') {
            ++position_;
            return Value(std::move(members));
        }
        while (true) {
            skipWhitespace();
            if (peek() != '"') {
                fail("khoá của object phải là chuỗi");
            }
            std::string key = parseString();
            skipWhitespace();
            if (peek() != ':') {
                fail("thiếu ':' sau khoá");
            }
            ++position_;
            skipWhitespace();
            members.emplace_back(std::move(key), parseValue());
            skipWhitespace();
            if (peek() == ',') {
                ++position_;
                continue;
            }
            if (peek() == '}') {
                ++position_;
                break;
            }
            fail("mong đợi ',' hoặc '}'");
        }
        return Value(std::move(members));
    }

    Value parseArray() {
        Array items;
        ++position_;  // '['
        skipWhitespace();
        if (peek() == ']') {
            ++position_;
            return Value(std::move(items));
        }
        while (true) {
            skipWhitespace();
            items.push_back(parseValue());
            skipWhitespace();
            if (peek() == ',') {
                ++position_;
                continue;
            }
            if (peek() == ']') {
                ++position_;
                break;
            }
            fail("mong đợi ',' hoặc ']'");
        }
        return Value(std::move(items));
    }

    std::string parseString() {
        ++position_;  // '"'
        std::string out;
        while (true) {
            if (position_ >= text_.size()) {
                fail("chuỗi chưa đóng");
            }
            const char c = text_[position_];
            if (c == '"') {
                ++position_;
                break;
            }
            if (c != '\\') {
                out.push_back(c);
                ++position_;
                continue;
            }
            ++position_;
            if (position_ >= text_.size()) {
                fail("escape chưa hoàn chỉnh");
            }
            const char escape = text_[position_++];
            switch (escape) {
                case '"':
                    out.push_back('"');
                    break;
                case '\\':
                    out.push_back('\\');
                    break;
                case '/':
                    out.push_back('/');
                    break;
                case 'b':
                    out.push_back('\b');
                    break;
                case 'f':
                    out.push_back('\f');
                    break;
                case 'n':
                    out.push_back('\n');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case 'u': {
                    unsigned int codePoint = parseHex4();
                    if (codePoint >= 0xD800 && codePoint <= 0xDBFF &&
                        position_ + 1 < text_.size() && text_[position_] == '\\' &&
                        text_[position_ + 1] == 'u') {
                        position_ += 2;
                        const unsigned int low = parseHex4();
                        codePoint = 0x10000 + ((codePoint - 0xD800) << 10) + (low - 0xDC00);
                    }
                    out += utf8::encode(codePoint);
                    break;
                }
                default:
                    fail("escape không hợp lệ");
            }
        }
        return out;
    }

    unsigned int parseHex4() {
        if (position_ + 4 > text_.size()) {
            fail("thiếu ký tự hex sau \\u");
        }
        unsigned int value = 0;
        for (int index = 0; index < 4; ++index) {
            const char c = text_[position_++];
            value <<= 4;
            if (c >= '0' && c <= '9') {
                value |= static_cast<unsigned int>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                value |= static_cast<unsigned int>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                value |= static_cast<unsigned int>(c - 'A' + 10);
            } else {
                fail("ký tự hex không hợp lệ");
            }
        }
        return value;
    }

    Value parseNumber() {
        const std::size_t start = position_;
        if (peek() == '-' || peek() == '+') {
            ++position_;
        }
        bool integral = true;
        while (position_ < text_.size()) {
            const char c = text_[position_];
            if (c >= '0' && c <= '9') {
                ++position_;
            } else if (c == '.' || c == 'e' || c == 'E') {
                integral = false;
                ++position_;
            } else if ((c == '-' || c == '+') && (text_[position_ - 1] == 'e' ||
                                                  text_[position_ - 1] == 'E')) {
                ++position_;
            } else {
                break;
            }
        }
        if (position_ == start) {
            fail("giá trị không hợp lệ");
        }
        const std::string token = text_.substr(start, position_ - start);
        if (integral) {
            return Value(static_cast<long long>(std::strtoll(token.c_str(), nullptr, 10)));
        }
        return Value(std::strtod(token.c_str(), nullptr));
    }

    const std::string& text_;
    std::size_t position_ = 0;
};

void escapeTo(std::string& out, const std::string& text) {
    out.push_back('"');
    for (const char c : text) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            default: {
                const auto byte = static_cast<unsigned char>(c);
                if (byte < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", byte);
                    out += buffer;
                } else {
                    out.push_back(c);  // UTF-8 giữ nguyên (ensure_ascii=False)
                }
                break;
            }
        }
    }
    out.push_back('"');
}

std::string formatNumber(double value, long long integer, bool integral) {
    if (integral) {
        return std::to_string(integer);
    }
    if (std::isnan(value) || std::isinf(value)) {
        return "null";
    }
    // %.10g giữ đủ chữ số cho các giá trị tiền tệ dạng 163636.3636 mà không
    // sinh đuôi nhiễu kiểu 163636.36360000001.
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.10g", value);
    return buffer;
}

}  // namespace

long long Value::asInteger(long long fallback) const {
    if (!isNumber()) {
        return fallback;
    }
    if (integral_) {
        return integer_;
    }
    return static_cast<long long>(number_);
}

const Value* Value::find(const std::string& key) const {
    if (type_ != Type::Object) {
        return nullptr;
    }
    for (const auto& member : object_) {
        if (member.first == key) {
            return &member.second;
        }
    }
    return nullptr;
}

std::vector<std::string> Value::asStringArray() const {
    std::vector<std::string> items;
    if (type_ != Type::Array) {
        return items;
    }
    for (const auto& item : array_) {
        if (item.isString()) {
            items.push_back(item.asString());
        }
    }
    return items;
}

Value Value::parse(const std::string& text) {
    Parser parser(text);
    return parser.parse();
}

Value Value::parseFile(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw JsonError("Không mở được file JSON: " + path);
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return parse(buffer.str());
}

std::string Value::dump(int indent) const {
    std::string out;
    dumpTo(out, indent, 0);
    return out;
}

void Value::dumpTo(std::string& out, int indent, int depth) const {
    const bool pretty = indent >= 0;
    const std::string pad = pretty ? std::string(static_cast<std::size_t>(indent) *
                                                     static_cast<std::size_t>(depth + 1),
                                                 ' ')
                                   : std::string();
    const std::string padClose =
        pretty ? std::string(static_cast<std::size_t>(indent) * static_cast<std::size_t>(depth),
                             ' ')
               : std::string();

    switch (type_) {
        case Type::Null:
            out += "null";
            break;
        case Type::Bool:
            out += boolean_ ? "true" : "false";
            break;
        case Type::Number:
            out += formatNumber(number_, integer_, integral_);
            break;
        case Type::String:
            escapeTo(out, text_);
            break;
        case Type::Array: {
            if (array_.empty()) {
                out += "[]";
                break;
            }
            out += '[';
            for (std::size_t index = 0; index < array_.size(); ++index) {
                if (index > 0) {
                    out += ',';
                }
                if (pretty) {
                    out += '\n';
                    out += pad;
                }
                array_[index].dumpTo(out, indent, depth + 1);
            }
            if (pretty) {
                out += '\n';
                out += padClose;
            }
            out += ']';
            break;
        }
        case Type::Object: {
            if (object_.empty()) {
                out += "{}";
                break;
            }
            out += '{';
            for (std::size_t index = 0; index < object_.size(); ++index) {
                if (index > 0) {
                    out += ',';
                }
                if (pretty) {
                    out += '\n';
                    out += pad;
                }
                escapeTo(out, object_[index].first);
                out += ':';
                if (pretty) {
                    out += ' ';
                }
                object_[index].second.dumpTo(out, indent, depth + 1);
            }
            if (pretty) {
                out += '\n';
                out += padClose;
            }
            out += '}';
            break;
        }
    }
}

}  // namespace ctkm::json
