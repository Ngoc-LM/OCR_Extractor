// Port của ``ctkm_extractor/extraction/schema.yaml`` + phần nạp schema trong
// ``extraction/extractor.py``. Bản C++ dùng JSON thay YAML để không phải kéo
// thêm thư viện phân tích YAML.
//
// Toàn bộ tri thức về "mẫu bảng" nằm trong file schema, KHÔNG nằm trong code:
// thêm field mới hoặc hỗ trợ mẫu CTKM khác chỉ cần sửa ``schema.json``.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "extraction/FieldParsers.hpp"
#include "util/Json.hpp"

namespace ctkm::extraction {

/// Schema không hợp lệ hoặc không đọc được.
class SchemaError : public std::runtime_error {
public:
    explicit SchemaError(const std::string& message) : std::runtime_error(message) {}
};

/// Khai báo của một field trong schema.
struct FieldSpec {
    std::string name;
    std::string parser = "text_parser";
    std::string type = "string";
    std::vector<std::string> aliases;
    std::vector<std::string> regex;
    json::Value parserArgs;
    bool required = false;

    ParserArgs args() const { return ParserArgs(parserArgs); }
    /// True nếu field khoanh vùng theo ``keyword`` (được loại khỏi collision guard).
    bool hasKeyword() const;
};

/// Tham số điều khiển việc so khớp nhãn.
struct SchemaSettings {
    /// Số hàng đầu bảng được coi là vùng header khi bảng nằm ngang.
    int headerRows = 3;
    /// Ngưỡng tương đồng tối thiểu (0..1) khi so khớp nhãn bằng fuzzy matching.
    double minSimilarity = 0.82;
    /// Số ô tối đa dò sang phải/xuống dưới để tìm ô giá trị của một nhãn.
    int maxValueDistance = 4;
};

/// Toàn bộ schema đã nạp.
struct Schema {
    std::vector<FieldSpec> fields;
    SchemaSettings settings;
    int version = 1;
    std::string path;

    std::vector<std::string> fieldNames() const;
};

/// Nạp schema từ file JSON.
/// @throws SchemaError khi file không tồn tại, JSON hỏng, hoặc thiếu "fields".
Schema loadSchema(const std::string& path = "");

/// Đường dẫn schema mặc định: biến môi trường CTKM_SCHEMA, ./schema.json,
/// cạnh file thực thi, hoặc đường dẫn cấu hình lúc build.
std::string defaultSchemaPath();

}  // namespace ctkm::extraction
