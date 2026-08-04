#include "extraction/Schema.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#include "util/Log.hpp"

namespace ctkm::extraction {
namespace {

constexpr const char* kLogger = "ctkm.extraction.schema";

#ifndef CTKM_DEFAULT_SCHEMA_PATH
#define CTKM_DEFAULT_SCHEMA_PATH ""
#endif

#ifndef CTKM_INSTALL_SCHEMA_RELATIVE_PATH
#define CTKM_INSTALL_SCHEMA_RELATIVE_PATH "../share/ctkm_extractor/schema.json"
#endif

/// Thư mục chứa file thực thi đang chạy; rỗng nếu không xác định được.
///
/// Cần thiết để bản đã `cmake --install` chạy được từ thư mục bất kỳ mà không
/// phải đặt CTKM_SCHEMA: schema nằm ở <prefix>/share/ctkm_extractor/ còn binary ở
/// <prefix>/bin/, và cây source lúc build có thể đã bị xoá/di chuyển.
std::filesystem::path executableDirectory() {
    std::error_code code;
#if defined(__linux__)
    const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", code);
    if (!code && !self.empty()) {
        return self.parent_path();
    }
#elif defined(__APPLE__)
    char buffer[4096];
    std::uint32_t size = sizeof(buffer);
    if (_NSGetExecutablePath(buffer, &size) == 0) {
        return std::filesystem::path(buffer).parent_path();
    }
#endif
    return std::filesystem::path();
}

std::vector<std::string> readStringList(const json::Value* value) {
    std::vector<std::string> items;
    if (value == nullptr) {
        return items;
    }
    if (value->isString()) {
        items.push_back(value->asString());
        return items;
    }
    if (value->isArray()) {
        return value->asStringArray();
    }
    return items;
}

}  // namespace

bool FieldSpec::hasKeyword() const {
    const json::Value* keyword = parserArgs.find("keyword");
    return keyword != nullptr && keyword->isString() && !keyword->asString().empty();
}

std::vector<std::string> Schema::fieldNames() const {
    std::vector<std::string> names;
    names.reserve(fields.size());
    for (const auto& spec : fields) {
        names.push_back(spec.name);
    }
    return names;
}

std::string defaultSchemaPath() {
    // 1. Biến môi trường CTKM_SCHEMA - ưu tiên cao nhất, cho phép ghi đè mọi thứ.
    if (const char* fromEnvironment = std::getenv("CTKM_SCHEMA")) {
        if (std::filesystem::is_regular_file(fromEnvironment)) {
            return fromEnvironment;
        }
    }
    // 2. Thư mục làm việc hiện tại.
    if (std::filesystem::is_regular_file("schema.json")) {
        return "schema.json";
    }
    // 3. Cạnh file thực thi (thư mục build) và vị trí sau khi `cmake --install`
    //    (<prefix>/bin/../share/ctkm_extractor/schema.json).
    const std::filesystem::path executableDir = executableDirectory();
    if (!executableDir.empty()) {
        for (const char* relative : {"schema.json", CTKM_INSTALL_SCHEMA_RELATIVE_PATH}) {
            std::error_code code;
            const std::filesystem::path candidate =
                std::filesystem::weakly_canonical(executableDir / relative, code);
            const std::filesystem::path resolved =
                code ? (executableDir / relative) : candidate;
            if (std::filesystem::is_regular_file(resolved)) {
                return resolved.string();
            }
        }
    }
    // 4. Cuối cùng mới tới cây source lúc build (chỉ còn khi chạy tại chỗ).
    const std::string configured = CTKM_DEFAULT_SCHEMA_PATH;
    if (!configured.empty() && std::filesystem::is_regular_file(configured)) {
        return configured;
    }
    return "schema.json";
}

Schema loadSchema(const std::string& path) {
    const std::string schemaPath = path.empty() ? defaultSchemaPath() : path;
    if (!std::filesystem::is_regular_file(schemaPath)) {
        throw SchemaError("Không tìm thấy schema: " + schemaPath);
    }

    json::Value payload;
    try {
        payload = json::Value::parseFile(schemaPath);
    } catch (const json::JsonError& error) {
        throw SchemaError("Không đọc được schema " + schemaPath + ": " + error.what());
    }
    if (!payload.isObject()) {
        throw SchemaError("Schema " + schemaPath + " phải là một object JSON");
    }

    const json::Value* rawFields = payload.find("fields");
    if (rawFields == nullptr || !rawFields->isArray() || rawFields->asArray().empty()) {
        throw SchemaError("Schema " + schemaPath + " thiếu danh sách 'fields'");
    }

    Schema schema;
    schema.path = schemaPath;
    if (const json::Value* version = payload.find("version")) {
        schema.version = static_cast<int>(version->asInteger(1));
    }
    if (const json::Value* settings = payload.find("settings")) {
        if (const json::Value* headerRows = settings->find("header_rows")) {
            schema.settings.headerRows = static_cast<int>(
                headerRows->asInteger(schema.settings.headerRows));
        }
        if (const json::Value* minSimilarity = settings->find("min_similarity")) {
            schema.settings.minSimilarity =
                minSimilarity->asDouble(schema.settings.minSimilarity);
        }
        if (const json::Value* maxDistance = settings->find("max_value_distance")) {
            schema.settings.maxValueDistance = static_cast<int>(
                maxDistance->asInteger(schema.settings.maxValueDistance));
        }
    }

    for (const auto& item : rawFields->asArray()) {
        if (!item.isObject()) {
            log::warn(kLogger, "Bỏ qua mục field không hợp lệ trong schema");
            continue;
        }
        const json::Value* name = item.find("name");
        if (name == nullptr || !name->isString() || name->asString().empty()) {
            log::warn(kLogger, "Bỏ qua field thiếu thuộc tính 'name'");
            continue;
        }
        FieldSpec spec;
        spec.name = name->asString();
        if (const json::Value* parser = item.find("parser")) {
            if (parser->isString() && !parser->asString().empty()) {
                spec.parser = parser->asString();
            }
        }
        if (const json::Value* type = item.find("type")) {
            if (type->isString() && !type->asString().empty()) {
                spec.type = type->asString();
            }
        }
        spec.aliases = readStringList(item.find("aliases"));
        spec.regex = readStringList(item.find("regex"));
        if (const json::Value* args = item.find("parser_args")) {
            if (args->isObject()) {
                spec.parserArgs = *args;
            }
        }
        if (const json::Value* required = item.find("required")) {
            spec.required = required->asBool(false);
        }
        schema.fields.push_back(std::move(spec));
    }

    if (schema.fields.empty()) {
        throw SchemaError("Schema " + schemaPath + " không có field hợp lệ nào");
    }
    return schema;
}

}  // namespace ctkm::extraction
