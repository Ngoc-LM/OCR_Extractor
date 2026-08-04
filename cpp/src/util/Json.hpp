// JSON tối giản, tự chứa (không phụ thuộc thư viện ngoài).
//
// Dùng cho 2 việc: đọc ``schema.json`` (tương đương ``schema.yaml`` bản Python)
// và ghi kết quả trích xuất. Object giữ nguyên THỨ TỰ khoá khi parse/dump để
// output JSON có thứ tự field đúng như khai báo schema, giống bản Python.
//
// Chuỗi được xuất ở dạng UTF-8 nguyên bản (không escape \uXXXX), tương đương
// ``json.dumps(..., ensure_ascii=False)``.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ctkm::json {

/// Lỗi cú pháp JSON khi parse.
class JsonError : public std::runtime_error {
public:
    explicit JsonError(const std::string& message) : std::runtime_error(message) {}
};

class Value;

using Array = std::vector<Value>;
using Member = std::pair<std::string, Value>;
using Object = std::vector<Member>;  // giữ thứ tự khoá

/// Một giá trị JSON bất kỳ.
class Value {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Value() = default;
    explicit Value(std::nullptr_t) {}
    explicit Value(bool value) : type_(Type::Bool), boolean_(value) {}
    explicit Value(double value)
        : type_(Type::Number), number_(value), integral_(false) {}
    explicit Value(long long value)
        : type_(Type::Number),
          number_(static_cast<double>(value)),
          integer_(value),
          integral_(true) {}
    explicit Value(int value) : Value(static_cast<long long>(value)) {}
    explicit Value(std::string value) : type_(Type::String), text_(std::move(value)) {}
    explicit Value(const char* value) : type_(Type::String), text_(value) {}
    explicit Value(Array value) : type_(Type::Array), array_(std::move(value)) {}
    explicit Value(Object value) : type_(Type::Object), object_(std::move(value)) {}

    Type type() const { return type_; }
    bool isNull() const { return type_ == Type::Null; }
    bool isBool() const { return type_ == Type::Bool; }
    bool isNumber() const { return type_ == Type::Number; }
    bool isString() const { return type_ == Type::String; }
    bool isArray() const { return type_ == Type::Array; }
    bool isObject() const { return type_ == Type::Object; }
    /// True nếu số được viết dưới dạng nguyên (không có phần thập phân/mũ).
    bool isIntegral() const { return type_ == Type::Number && integral_; }

    bool asBool(bool fallback = false) const { return isBool() ? boolean_ : fallback; }
    double asDouble(double fallback = 0.0) const { return isNumber() ? number_ : fallback; }
    long long asInteger(long long fallback = 0) const;
    const std::string& asString() const { return text_; }
    const Array& asArray() const { return array_; }
    const Object& asObject() const { return object_; }

    /// Giá trị của khoá trong object, ``nullptr`` nếu không có.
    const Value* find(const std::string& key) const;
    /// Có khoá này không (chỉ đúng với object).
    bool contains(const std::string& key) const { return find(key) != nullptr; }

    /// Danh sách chuỗi; phần tử không phải chuỗi bị bỏ qua.
    std::vector<std::string> asStringArray() const;

    /// Parse chuỗi JSON; ném :class:`JsonError` khi cú pháp sai.
    static Value parse(const std::string& text);
    /// Đọc và parse một file JSON; ném :class:`JsonError` khi lỗi.
    static Value parseFile(const std::string& path);

    /// Serialise; ``indent < 0`` là dạng gọn một dòng.
    std::string dump(int indent = -1) const;

private:
    void dumpTo(std::string& out, int indent, int depth) const;

    Type type_ = Type::Null;
    bool boolean_ = false;
    double number_ = 0.0;
    long long integer_ = 0;
    bool integral_ = false;
    std::string text_;
    Array array_;
    Object object_;
};

}  // namespace ctkm::json
