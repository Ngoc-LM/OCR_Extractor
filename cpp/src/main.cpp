// Port của ``ctkm_extractor/cli.py``.
//
//   ./ctkm_extractor --image <path> --out result.json [--debug]
//
// Chương trình luôn ghi ra JSON đủ field (giá trị thiếu là null) kể cả khi OCR
// lỗi, và trả exit code khác 0 chỉ khi lỗi cấu hình (schema hỏng, không tìm thấy
// ảnh, không có engine nào khả dụng).

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "extraction/Extractor.hpp"
#include "extraction/Schema.hpp"
#include "ocr/OCRProvider.hpp"
#include "ocr/ProviderFactory.hpp"
#include "util/Log.hpp"

namespace {

constexpr int kExitOk = 0;
constexpr int kExitError = 1;
constexpr const char* kLogger = "ctkm_extractor";

/// Tham số dòng lệnh đã phân tích.
struct Arguments {
    std::vector<std::string> images;
    std::string textFile;
    std::string out;
    std::string engine = "paddle_vietocr";
    std::string schema;
    bool noMorphology = false;
    bool noPpStructure = false;
    bool strictEngine = false;
    bool noBinarize = false;
    bool binarize = false;
    bool debug = false;
    bool listEngines = false;
    bool help = false;
    int indent = 2;
};

void printUsage() {
    std::cout
        << "Trích xuất thông tin CTKM từ ảnh bảng biểu ra JSON.\n\n"
        << "Cách dùng:\n"
        << "  ctkm_extractor --image <path> [--out result.json] [tuỳ chọn]\n"
        << "  ctkm_extractor --text-file <path> [--out result.json] [tuỳ chọn]\n\n"
        << "Tuỳ chọn:\n"
        << "  --image <path>       Ảnh chứa bảng CTKM. LẶP LẠI ĐƯỢC: mỗi lần một\n"
           "                       trang (VD từng trang PDF đã render sẵn); chương\n"
           "                       trình OCR từng trang rồi gộp kết quả\n"
        << "  --text-file <path>   Bỏ qua OCR, đọc thẳng raw text từ file (debug tầng "
           "trích xuất)\n"
        << "  --out <path>         File JSON đầu ra; bỏ trống thì in ra stdout\n"
        << "  --engine <name>      paddle_vietocr (mặc định) | tesseract | auto\n"
        << "  --schema <path>      schema.json tuỳ biến; mặc định dùng schema đi kèm\n"
        << "  --no-morphology      Bỏ qua dò bảng bằng đường kẻ (CV cổ điển)\n"
        << "  --no-pp-structure    Bỏ qua PP-Structure\n"
        << "  --strict-engine      Không tự fallback sang engine khác\n"
        << "  --no-binarize        Ép TẮT adaptive threshold. Mặc định chương trình\n"
           "                       chạy CẢ HAI cấu hình rồi giữ kết quả trích được\n"
           "                       nhiều field hơn; ép để chỉ chạy một lượt\n"
        << "  --binarize           Ép BẬT adaptive threshold, chỉ chạy một lượt\n"
        << "  --indent <n>         Số space thụt lề JSON (mặc định 2)\n"
        << "  --debug              In OCR raw text, bảng đã dựng và nguồn từng field ra "
           "stderr\n"
        << "  --list-engines       In danh sách engine khả dụng rồi thoát\n"
        << "  -h, --help           Hiện trợ giúp này\n";
}

/// Đọc giá trị đi kèm một cờ; trả false nếu thiếu.
bool takeValue(const std::vector<std::string>& argv, std::size_t& index, std::string& target) {
    if (index + 1 >= argv.size()) {
        std::cerr << "Thiếu giá trị cho tham số " << argv[index] << '\n';
        return false;
    }
    target = argv[index + 1];
    ++index;
    return true;
}

bool parseArguments(const std::vector<std::string>& argv, Arguments& args) {
    for (std::size_t index = 0; index < argv.size(); ++index) {
        const std::string& flag = argv[index];
        if (flag == "--image") {
            // Lặp lại được: mỗi lần thêm một ảnh (VD từng trang PDF đã render).
            std::string value;
            if (!takeValue(argv, index, value)) {
                return false;
            }
            args.images.push_back(value);
        } else if (flag == "--text-file") {
            if (!takeValue(argv, index, args.textFile)) {
                return false;
            }
        } else if (flag == "--out") {
            if (!takeValue(argv, index, args.out)) {
                return false;
            }
        } else if (flag == "--engine") {
            if (!takeValue(argv, index, args.engine)) {
                return false;
            }
        } else if (flag == "--schema") {
            if (!takeValue(argv, index, args.schema)) {
                return false;
            }
        } else if (flag == "--indent") {
            std::string value;
            if (!takeValue(argv, index, value)) {
                return false;
            }
            try {
                args.indent = std::stoi(value);
            } catch (const std::exception&) {
                std::cerr << "Giá trị --indent không hợp lệ: " << value << '\n';
                return false;
            }
        } else if (flag == "--no-morphology") {
            args.noMorphology = true;
        } else if (flag == "--no-pp-structure") {
            args.noPpStructure = true;
        } else if (flag == "--strict-engine") {
            args.strictEngine = true;
        } else if (flag == "--no-binarize") {
            args.noBinarize = true;
        } else if (flag == "--binarize") {
            args.binarize = true;
        } else if (flag == "--debug") {
            args.debug = true;
        } else if (flag == "--list-engines") {
            args.listEngines = true;
        } else if (flag == "-h" || flag == "--help") {
            args.help = true;
        } else {
            std::cerr << "Tham số không hợp lệ: " << flag << '\n';
            return false;
        }
    }
    return true;
}

/// Ghi JSON ra file hoặc stdout.
bool writeOutput(const ctkm::extraction::ExtractionResult& result, const std::string& outPath,
                 int indent) {
    const std::string payload = result.toJsonString(indent);
    if (outPath.empty()) {
        std::cout << payload << '\n';
        return true;
    }
    const std::filesystem::path target(outPath);
    if (target.has_parent_path() && !target.parent_path().empty() &&
        !std::filesystem::exists(target.parent_path())) {
        std::error_code code;
        std::filesystem::create_directories(target.parent_path(), code);
    }
    std::ofstream stream(target, std::ios::binary);
    if (!stream) {
        ctkm::log::error(kLogger, "Không ghi được file kết quả: " + outPath);
        return false;
    }
    stream << payload << '\n';
    ctkm::log::info(kLogger, "Đã ghi kết quả vào " + outPath);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> arguments(argv + 1, argv + argc);

    Arguments args;
    if (!parseArguments(arguments, args)) {
        return kExitError;
    }
    if (args.help) {
        printUsage();
        return kExitOk;
    }

    ctkm::log::setLevel(args.debug ? ctkm::log::Level::Debug : ctkm::log::Level::Info);

    if (args.listEngines) {
        const auto engines = ctkm::ocr::availableEngines();
        std::cout << "Engine khả dụng: ";
        if (engines.empty()) {
            std::cout << "(không có)";
        } else {
            for (std::size_t index = 0; index < engines.size(); ++index) {
                if (index > 0) {
                    std::cout << ", ";
                }
                std::cout << engines[index];
            }
        }
        std::cout << '\n';
        return kExitOk;
    }

    if (args.images.empty() == args.textFile.empty()) {
        std::cerr << "Cần đúng một trong hai tham số --image hoặc --text-file\n\n";
        printUsage();
        return kExitError;
    }

    ctkm::extraction::CTKMExtractor::Options options;
    options.engine = args.engine;
    options.useMorphology = !args.noMorphology;
    options.usePpStructure = !args.noPpStructure;
    options.strictEngine = args.strictEngine;
    // Rỗng = tự chọn: chạy cả hai cấu hình rồi giữ kết quả nhiều field hơn.
    if (args.noBinarize) {
        options.binarize = false;
    } else if (args.binarize) {
        options.binarize = true;
    }

    std::unique_ptr<ctkm::extraction::CTKMExtractor> extractor;
    try {
        extractor = std::make_unique<ctkm::extraction::CTKMExtractor>(args.schema, options);
    } catch (const ctkm::extraction::SchemaError& error) {
        ctkm::log::error(kLogger, std::string("Schema lỗi: ") + error.what());
        return kExitError;
    }

    ctkm::extraction::ExtractionResult result;
    try {
        if (!args.textFile.empty()) {
            std::ifstream stream(args.textFile, std::ios::binary);
            if (!stream) {
                ctkm::log::error(kLogger, "Không đọc được file text: " + args.textFile);
                return kExitError;
            }
            std::ostringstream buffer;
            buffer << stream.rdbuf();
            result = extractor->extractFromText(buffer.str());
        } else {
            for (const auto& path : args.images) {
                if (std::filesystem::is_regular_file(path)) {
                    continue;
                }
                ctkm::log::error(kLogger, "Không tìm thấy ảnh: " + path);
                return kExitError;
            }
            result = args.images.size() == 1
                         ? extractor->extractFromImage(args.images.front())
                         : extractor->extractFromImages(args.images);
        }
    } catch (const ctkm::ocr::ProviderUnavailableError& error) {
        ctkm::log::error(kLogger,
                         std::string("Không khởi tạo được OCR engine (") + error.what() +
                             "). Đặt models/det.onnx + models/vietocr.onnx, hoặc chạy lại "
                             "với --engine tesseract.");
        return kExitError;
    } catch (const std::exception& error) {
        ctkm::log::error(kLogger, std::string("Lỗi khi xử lý đầu vào: ") + error.what());
        return kExitError;
    }

    if (args.debug) {
        std::cerr << result.debugReport();
    }

    if (!writeOutput(result, args.out, args.indent)) {
        return kExitError;
    }

    std::string missing;
    for (const auto& field : result.fields) {
        if (!field.found()) {
            if (!missing.empty()) {
                missing += ", ";
            }
            missing += field.name;
        }
    }
    if (!missing.empty()) {
        ctkm::log::warn(kLogger, "Các field không trích xuất được (null): " + missing);
    }
    return kExitOk;
}
