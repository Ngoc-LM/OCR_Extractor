#include "extraction/Extractor.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <functional>
#include <cmath>
#include <iomanip>
#include <map>
#include <set>
#include <regex>
#include <sstream>

#include "ocr/ProviderFactory.hpp"
#include "util/Log.hpp"
#include "util/Utf8.hpp"

namespace ctkm::extraction {

const char* const kSourceTable = "table";
const char* const kSourceTextLine = "text_line";
const char* const kSourceRegex = "regex";
const char* const kSourceMissing = "missing";

namespace {

constexpr const char* kLogger = "ctkm.extraction.extractor";

std::string formatScore(double score) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << score;
    return stream.str();
}

std::vector<std::string> splitLines(const std::string& text) {
    std::vector<std::string> lines;
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.push_back(line);
    }
    return lines;
}

/// Port ``difflib.SequenceMatcher.find_longest_match`` (không có junk).
struct Match {
    std::size_t a = 0;
    std::size_t b = 0;
    std::size_t size = 0;
};

Match findLongestMatch(const std::string& a,
                       const std::map<char, std::vector<std::size_t>>& b2j, std::size_t alo,
                       std::size_t ahi, std::size_t blo, std::size_t bhi) {
    Match best{alo, blo, 0};
    std::map<std::size_t, std::size_t> j2len;
    for (std::size_t i = alo; i < ahi; ++i) {
        std::map<std::size_t, std::size_t> newJ2len;
        const auto found = b2j.find(a[i]);
        if (found != b2j.end()) {
            for (const std::size_t j : found->second) {
                if (j < blo) {
                    continue;
                }
                if (j >= bhi) {
                    break;
                }
                std::size_t previous = 0;
                if (j > 0) {
                    const auto entry = j2len.find(j - 1);
                    if (entry != j2len.end()) {
                        previous = entry->second;
                    }
                }
                const std::size_t k = previous + 1;
                newJ2len[j] = k;
                if (k > best.size) {
                    best = Match{i - k + 1, j - k + 1, k};
                }
            }
        }
        j2len = std::move(newJ2len);
    }
    return best;
}

}  // namespace

const std::vector<std::string>& labelValueSeparators() {
    static const std::vector<std::string> separators = {":", "-", "–", "—", "|", "="};
    return separators;
}

double similarity(const std::string& left, const std::string& right) {
    if (left.empty() || right.empty()) {
        return 0.0;
    }
    std::map<char, std::vector<std::size_t>> b2j;
    for (std::size_t index = 0; index < right.size(); ++index) {
        b2j[right[index]].push_back(index);
    }

    std::size_t matched = 0;
    std::vector<std::array<std::size_t, 4>> queue{{0, left.size(), 0, right.size()}};
    while (!queue.empty()) {
        const auto range = queue.back();
        queue.pop_back();
        const Match match =
            findLongestMatch(left, b2j, range[0], range[1], range[2], range[3]);
        if (match.size == 0) {
            continue;
        }
        matched += match.size;
        if (range[0] < match.a && range[2] < match.b) {
            queue.push_back({range[0], match.a, range[2], match.b});
        }
        if (match.a + match.size < range[1] && match.b + match.size < range[3]) {
            queue.push_back({match.a + match.size, range[1], match.b + match.size, range[3]});
        }
    }
    const double total = static_cast<double>(left.size() + right.size());
    if (total <= 0.0) {
        return 0.0;
    }
    return 2.0 * static_cast<double>(matched) / total;
}

AliasMatch matchAlias(const std::string& text, const std::vector<std::string>& aliases,
                      double minSimilarity) {
    AliasMatch result;
    const std::string normalizedText = normalizeLabel(text);
    if (normalizedText.empty()) {
        return result;
    }
    double bestScore = 0.0;
    std::optional<std::string> bestAlias;
    for (const auto& alias : aliases) {
        const std::string normalizedAlias = normalizeLabel(alias);
        if (normalizedAlias.empty()) {
            continue;
        }
        double score = 0.0;
        if (normalizedText == normalizedAlias) {
            score = 1.0;
        } else if (normalizedText.rfind(normalizedAlias, 0) == 0) {
            score = 0.97;
        } else if (normalizedText.find(normalizedAlias) != std::string::npos) {
            // Nhãn nằm lẫn trong câu dài thì độ tin cậy giảm theo tỉ lệ độ dài.
            const double ratio = static_cast<double>(normalizedAlias.size()) /
                                 static_cast<double>(normalizedText.size());
            score = 0.85 + 0.1 * ratio;
        } else {
            score = similarity(normalizedText, normalizedAlias);
        }
        if (score > bestScore) {
            bestScore = score;
            bestAlias = alias;
        }
    }
    result.score = bestScore;
    if (bestScore >= minSimilarity) {
        result.alias = bestAlias;
    }
    return result;
}

std::string remainderAfterAlias(const std::string& text, const std::string& alias) {
    if (alias.empty()) {
        return std::string();
    }
    const FoldedText folded = foldAccents(text);
    std::string haystack = folded.folded;
    std::transform(haystack.begin(), haystack.end(), haystack.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::string needle = foldAccentsSimple(alias);
    std::transform(needle.begin(), needle.end(), needle.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (needle.empty()) {
        return std::string();
    }
    const std::size_t position = haystack.find(needle);
    if (position == std::string::npos) {
        return std::string();
    }
    const std::string tail = text.substr(folded.originalOffset(position + needle.size()));
    return trim(stripCharacters(trim(tail), labelValueSeparators()));
}

namespace {

std::string lowerAscii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

/// True nếu ``folded[start, start+length)`` là một cụm từ trọn vẹn.
bool isWordBoundary(const std::string& folded, std::size_t start, std::size_t length) {
    const unsigned char before = start > 0 ? static_cast<unsigned char>(folded[start - 1]) : ' ';
    const std::size_t afterIndex = start + length;
    const unsigned char after =
        afterIndex < folded.size() ? static_cast<unsigned char>(folded[afterIndex]) : ' ';
    return std::isalnum(before) == 0 && std::isalnum(after) == 0;
}

}  // namespace

std::string cutAtNextLabel(const std::string& value,
                           const std::vector<std::string>& ownAliases,
                           const std::vector<std::string>& otherAliases) {
    if (value.empty()) {
        return value;
    }
    const FoldedText folded = foldAccents(value);
    const std::string haystack = lowerAscii(folded.folded);

    std::set<std::string> own;
    for (const auto& alias : ownAliases) {
        own.insert(lowerAscii(foldAccentsSimple(alias)));
    }

    std::size_t cut = haystack.size();
    for (const auto& alias : otherAliases) {
        const std::string needle = trim(lowerAscii(foldAccentsSimple(alias)));
        if (needle.size() < kMinBoundaryAliasLength || own.count(needle) > 0) {
            continue;
        }
        std::size_t position = haystack.find(needle);
        while (position != std::string::npos && position < cut) {
            if (isWordBoundary(haystack, position, needle.size())) {
                cut = position;
                break;
            }
            position = haystack.find(needle, position + 1);
        }
    }
    if (cut >= haystack.size()) {
        return value;
    }
    const std::string head = value.substr(0, folded.originalOffset(cut));
    return trim(stripCharacters(trim(head), labelValueSeparators()));
}

// ---------------------------------------------------------------------------
// ExtractionResult
// ---------------------------------------------------------------------------
const FieldResult* ExtractionResult::field(const std::string& name) const {
    for (const auto& item : fields) {
        if (item.name == name) {
            return &item;
        }
    }
    return nullptr;
}

json::Value ExtractionResult::toJson() const {
    json::Object object;
    for (const auto& item : fields) {
        object.emplace_back(item.name, ctkm::extraction::toJson(item.value));
    }
    return json::Value(std::move(object));
}

std::string ExtractionResult::toJsonString(int indent) const { return toJson().dump(indent); }

std::string ExtractionResult::debugReport() const {
    std::ostringstream out;
    out << "OCR provider     : " << ocrProvider << '\n';
    out << "Table strategy   : " << tableStrategy << '\n';
    if (pagesProcessed > 0) {
        out << "Trang            : " << (page.has_value() ? std::to_string(*page) : "-")
            << " (chính) trong " << pagesProcessed << " trang đã OCR" << '\n';
    }
    if (table.has_value()) {
        out << "Table size       : " << table->rowCount() << " hàng x " << table->columnCount()
            << " cột (confidence=" << formatScore(table->confidence()) << ")\n";
    }
    out << '\n';
    out << "--- OCR raw text ---\n";
    out << (rawText.empty() ? std::string("(rỗng)") : rawText) << '\n';
    out << '\n';
    out << "--- Bảng đã dựng ---\n";
    out << (table.has_value() ? table->render() : std::string("(không dựng được bảng)")) << '\n';
    out << '\n';
    out << "--- Nguồn từng field ---\n";
    for (const auto& item : fields) {
        std::string name = item.name;
        if (name.size() < 16) {
            name.append(16 - name.size(), ' ');
        }
        std::string value = describe(item.value);
        if (value.size() < 28) {
            value.append(28 - value.size(), ' ');
        }
        out << name << " = " << value << " [source=" << item.source
            << ", score=" << formatScore(item.score) << ", raw="
            << (item.rawValue.has_value() ? "'" + *item.rawValue + "'" : "None") << "]\n";
    }
    if (!warnings.empty()) {
        out << '\n';
        out << "--- Cảnh báo ---\n";
        for (const auto& warning : warnings) {
            out << "* " << warning << '\n';
        }
    }
    return out.str();
}

// ---------------------------------------------------------------------------
// CTKMExtractor
// ---------------------------------------------------------------------------
CTKMExtractor::CTKMExtractor(Schema schema, Options options)
    : schema_(std::move(schema)), options_(std::move(options)) {}

CTKMExtractor::CTKMExtractor(const std::string& schemaPath, Options options)
    : schema_(loadSchema(schemaPath)), options_(std::move(options)) {}

ocr::OCRProvider& CTKMExtractor::provider() {
    if (!provider_) {
        provider_ = ocr::createProvider(options_.engine, options_.strictEngine,
                                       options_.provider);
    }
    return *provider_;
}

std::pair<int, double> pageExtractionScore(const ExtractionResult& result) {
    int found = 0;
    double total = 0.0;
    for (const auto& item : result.fields) {
        if (item.found()) {
            ++found;
            total += item.score;
        }
    }
    if (found == 0) {
        return {0, 0.0};
    }
    return {found, total / static_cast<double>(found)};
}

ExtractionResult mergePageResults(
    const std::vector<std::pair<int, ExtractionResult>>& pages) {
    ExtractionResult merged;
    if (pages.empty()) {
        return merged;
    }

    std::vector<std::size_t> order(pages.size());
    for (std::size_t index = 0; index < pages.size(); ++index) {
        order[index] = index;
    }
    std::stable_sort(order.begin(), order.end(), [&pages](std::size_t a, std::size_t b) {
        const auto scoreA = pageExtractionScore(pages[a].second);
        const auto scoreB = pageExtractionScore(pages[b].second);
        if (scoreA.first != scoreB.first) {
            return scoreA.first > scoreB.first;
        }
        if (scoreA.second != scoreB.second) {
            return scoreA.second > scoreB.second;
        }
        return pages[a].first < pages[b].first;   // hoà thì lấy trang số nhỏ hơn
    });

    const int primaryPage = pages[order.front()].first;
    const ExtractionResult& primary = pages[order.front()].second;

    merged = primary;
    merged.page = primaryPage;
    merged.pagesProcessed = static_cast<int>(pages.size());
    for (auto& item : merged.fields) {
        item.sourcePage = item.found() ? std::optional<int>(primaryPage) : std::nullopt;
    }

    for (std::size_t rank = 1; rank < order.size(); ++rank) {
        const int pageNumber = pages[order[rank]].first;
        const ExtractionResult& other = pages[order[rank]].second;
        for (const auto& candidate : other.fields) {
            if (!candidate.found()) {
                continue;
            }
            for (auto& target : merged.fields) {
                if (target.name != candidate.name || target.found()) {
                    continue;
                }
                target = candidate;
                target.sourcePage = pageNumber;
                const std::string message =
                    "Field '" + candidate.name + "' lấy từ trang " +
                    std::to_string(pageNumber) + " (trang chính là " +
                    std::to_string(primaryPage) + ")";
                log::info(kLogger, message);
                merged.warnings.push_back(message);
                break;
            }
        }
    }
    return merged;
}

ExtractionResult CTKMExtractor::extractFromImages(
    const std::vector<std::string>& imagePaths) {
    std::vector<std::pair<int, ExtractionResult>> pages;
    pages.reserve(imagePaths.size());
    for (std::size_t index = 0; index < imagePaths.size(); ++index) {
        ExtractionResult result = extractFromImage(imagePaths[index]);
        const int number = static_cast<int>(index) + 1;
        log::info(kLogger, "Trang " + std::to_string(number) + " (" + imagePaths[index] +
                               "): trích được " +
                               std::to_string(pageExtractionScore(result).first) + " field");
        pages.emplace_back(number, std::move(result));
    }
    ExtractionResult merged = mergePageResults(pages);
    if (merged.page.has_value()) {
        log::info(kLogger, "Chọn trang " + std::to_string(*merged.page) +
                               " làm trang chính trong " + std::to_string(pages.size()) +
                               " trang");
    }
    return merged;
}

ExtractionResult CTKMExtractor::extractImageOnce(
    const std::string& imagePath, const std::optional<ocr::PreprocessConfig>& config) {
    std::vector<std::string> warnings;
    ocr::OCRResult ocrResult;
    if (config.has_value()) {
        provider().setPreprocess(*config);
    }
    try {
        ocrResult = provider().extract(imagePath);
    } catch (const std::exception& error) {
        const std::string message =
            "OCR thất bại trên " + imagePath + ": " + error.what();
        log::warn(kLogger, message);
        warnings.push_back(message);
        ocrResult = ocr::OCRResult{};
        ocrResult.imagePath = imagePath;
        ocrResult.provider = "unavailable";
    }
    ExtractionResult result = extractFromOcr(ocrResult);
    result.warnings.insert(result.warnings.begin(), warnings.begin(), warnings.end());
    return result;
}

ExtractionResult CTKMExtractor::extractFromImage(const std::string& imagePath) {
    if (options_.binarize.has_value()) {
        ocr::PreprocessConfig config = options_.provider.tesseract.preprocess;
        config.adaptiveThreshold = *options_.binarize;
        return extractImageOnce(imagePath, config);
    }

    // TỰ CHỌN: nhị phân hoá CẦN cho Tesseract nhưng biến nét watermark mờ thành
    // nét đen đặc đè lên chữ; detector DB của PaddleOCR thì ngược lại. Hai engine
    // cho kết quả trái ngược nhau trên cùng một ảnh, nên ép sẵn một bên là bắt
    // người dùng phải đoán. Chạy cả hai rồi giữ kết quả nhiều field hơn.
    ocr::PreprocessConfig probe = options_.provider.tesseract.preprocess;
    if (!provider().setPreprocess(probe)) {
        // Provider không có bước tiền xử lý (mock trong test) -> một lượt là đủ.
        return extractImageOnce(imagePath, std::nullopt);
    }

    ExtractionResult best;
    std::pair<int, double> bestScore{-1, -1.0};
    std::string bestLabel;
    for (const bool binarize : {true, false}) {
        ocr::PreprocessConfig config = probe;
        config.adaptiveThreshold = binarize;
        ExtractionResult candidate = extractImageOnce(imagePath, config);
        const auto score = pageExtractionScore(candidate);
        const std::string label = binarize ? "nhị phân hoá" : "không nhị phân hoá";
        log::info(kLogger, "Tiền xử lý " + label + ": trích được " +
                               std::to_string(score.first) + " field");
        if (score > bestScore) {
            bestScore = score;
            bestLabel = label;
            best = std::move(candidate);
        }
    }
    log::info(kLogger, "Chọn tiền xử lý '" + bestLabel + "' (" +
                           std::to_string(bestScore.first) + " field)");
    return best;
}

ExtractionResult CTKMExtractor::extractFromOcr(const ocr::OCRResult& ocrResult) {
    table::BuildOptions buildOptions;
    buildOptions.useMorphology = options_.useMorphology;
    buildOptions.usePpStructure = options_.usePpStructure;
    const table::TableBuildResult buildResult = table::buildTable(ocrResult, buildOptions);
    return extract(buildResult, ocrResult.provider, ocrResult.warnings);
}

ExtractionResult CTKMExtractor::extractFromText(const std::string& text) {
    const ocr::OCRResult ocrResult = ocr::OCRResult::fromText(text, "text");
    table::BuildOptions buildOptions;
    buildOptions.useMorphology = false;
    buildOptions.usePpStructure = false;
    const table::TableBuildResult buildResult = table::buildTable(ocrResult, buildOptions);
    return extract(buildResult, "text", {});
}

ExtractionResult CTKMExtractor::extractFromTable(const table::Table& tableValue,
                                                 const std::string& rawText) {
    table::TableBuildResult buildResult;
    buildResult.table = tableValue;
    buildResult.strategy = tableValue.source();
    buildResult.rawText = rawText;
    return extract(buildResult, "table", {});
}

ExtractionResult CTKMExtractor::extract(const table::TableBuildResult& buildResult,
                                        const std::string& providerName,
                                        const std::vector<std::string>& extraWarnings) {
    ExtractionResult result;
    result.rawText = buildResult.rawText;
    result.table = buildResult.table;
    result.tableStrategy = buildResult.strategy;
    result.ocrProvider = providerName;
    result.warnings = extraWarnings;
    result.warnings.insert(result.warnings.end(), buildResult.warnings.begin(),
                           buildResult.warnings.end());

    for (const auto& spec : schema_.fields) {
        FieldResult fieldResult = resolveField(spec, buildResult);
        if (!fieldResult.found()) {
            const std::string message =
                "Không trích xuất được field '" + spec.name + "' - trả null";
            if (spec.required) {
                log::warn(kLogger, message);
            } else {
                log::info(kLogger, message);
            }
            result.warnings.push_back(message);
        }
        result.fields.push_back(std::move(fieldResult));
    }

    const std::vector<std::string> collisions = resolveCellCollisions(result.fields);
    result.warnings.insert(result.warnings.end(), collisions.begin(), collisions.end());
    return result;
}

std::vector<std::string> CTKMExtractor::resolveCellCollisions(
    std::vector<FieldResult>& results) const {
    // Field khai báo ``keyword`` (VD youtubeGB/spotifyGB cùng đọc từ 1 ô gộp
    // nhưng mỗi field tự khoanh vùng theo keyword riêng) được LOẠI TRỪ khỏi kiểm
    // tra này - đó là thiết kế có chủ đích, không phải xung đột.
    std::vector<std::string> keyworded;
    for (const auto& spec : schema_.fields) {
        if (spec.hasKeyword()) {
            keyworded.push_back(spec.name);
        }
    }

    std::map<std::pair<int, int>, std::vector<FieldResult*>> byCell;
    for (auto& item : results) {
        if (item.source != kSourceTable || !item.cellCoords.has_value()) {
            continue;
        }
        if (std::find(keyworded.begin(), keyworded.end(), item.name) != keyworded.end()) {
            continue;
        }
        byCell[*item.cellCoords].push_back(&item);
    }

    std::vector<std::string> warnings;
    for (auto& entry : byCell) {
        if (entry.second.size() < 2) {
            continue;
        }
        std::stable_sort(entry.second.begin(), entry.second.end(),
                         [](const FieldResult* a, const FieldResult* b) {
                             return a->score > b->score;
                         });
        const FieldResult* winner = entry.second.front();
        for (std::size_t index = 1; index < entry.second.size(); ++index) {
            FieldResult* loser = entry.second[index];
            const std::string message =
                "Field '" + loser->name + "' và '" + winner->name + "' cùng khớp ô (" +
                std::to_string(entry.first.first) + ", " + std::to_string(entry.first.second) +
                ") (score " + formatScore(loser->score) + " vs " + formatScore(winner->score) +
                ") - giữ '" + winner->name + "', trả null cho '" + loser->name + "'";
            log::warn(kLogger, message);
            warnings.push_back(message);
            loser->value.reset();
            loser->source = kSourceMissing;
        }
    }
    return warnings;
}

FieldResult CTKMExtractor::resolveField(const FieldSpec& spec,
                                        const table::TableBuildResult& buildResult) const {
    std::vector<Candidate> candidates;

    if (buildResult.table.has_value()) {
        for (auto& candidate : findInTable(spec, *buildResult.table)) {
            candidates.push_back(std::move(candidate));
        }
    }

    const auto lineCandidate = findInTextLines(spec, buildResult.rawText);
    if (lineCandidate.has_value()) {
        candidates.push_back(*lineCandidate);
    }

    const auto regexValue = findByRegex(spec, buildResult.rawText);
    if (regexValue.has_value()) {
        Candidate candidate;
        candidate.source = kSourceRegex;
        candidate.rawValue = *regexValue;
        candidate.score = 0.7;
        candidates.push_back(std::move(candidate));
    }

    for (const auto& candidate : candidates) {
        const auto parsed = parseValue(spec.parser, candidate.rawValue, spec.args());
        const auto value = coerceType(spec, parsed);
        if (value.has_value()) {
            FieldResult result;
            result.name = spec.name;
            result.value = value;
            result.rawValue = candidate.rawValue;
            result.source = candidate.source;
            result.score = candidate.score;
            result.matchedAlias = candidate.alias;
            if (candidate.source == kSourceTable) {
                result.cellCoords = candidate.cellCoords;
            }
            return result;
        }
        log::debug(kLogger, "Field " + spec.name + ": parser " + spec.parser +
                                " không parse được '" + candidate.rawValue + "' (source=" +
                                candidate.source + ")");
    }

    FieldResult result;
    result.name = spec.name;
    result.source = kSourceMissing;
    if (!candidates.empty()) {
        result.rawValue = candidates.front().rawValue;
    }
    return result;
}

std::vector<CTKMExtractor::Candidate> CTKMExtractor::findInTable(
    const FieldSpec& spec, const table::Table& tableValue) const {
    std::vector<Candidate> candidates;

    const table::TableCell* bestCell = nullptr;
    double bestScore = 0.0;
    std::string bestAlias;
    for (const auto* cell : tableValue.sortedCells()) {
        if (cell->isEmpty()) {
            continue;
        }
        const AliasMatch match =
            matchAlias(cell->text, spec.aliases, schema_.settings.minSimilarity);
        if (!match.alias.has_value()) {
            continue;
        }
        if (bestCell == nullptr || match.score > bestScore) {
            bestCell = cell;
            bestScore = match.score;
            bestAlias = *match.alias;
        }
    }
    if (bestCell == nullptr) {
        return candidates;
    }

    const int row = bestCell->row;
    const int col = bestCell->col;
    const int spanRows = bestCell->rowSpan;
    const int spanCols = bestCell->colSpan;

    // Nhãn và giá trị nằm chung một ô: "Chu kỳ gia hạn: tháng".
    const std::string inline_ = remainderAfterAlias(bestCell->text, bestAlias);
    std::optional<Candidate> inlineCandidate;
    if (!inline_.empty()) {
        Candidate candidate;
        candidate.source = kSourceTable;
        candidate.rawValue = inline_;
        candidate.score = bestScore;
        candidate.alias = bestAlias;
        candidate.cellCoords = std::make_pair(row, col);
        inlineCandidate = candidate;
    }

    // Ô bên phải (bảng dạng nhãn - giá trị) và ô bên dưới (bảng có header trên
    // cùng). Chỉ coi là "hàng header" khi bảng THỰC SỰ nhiều cột: với bảng 1 cột
    // dựng từ text thô (mỗi dòng = 1 ô), "ô bên dưới" luôn tồn tại một cách tầm
    // thường nên không phải tín hiệu đáng tin để suy ra đây là header.
    const bool inHeader =
        row < std::max(1, schema_.settings.headerRows) && tableValue.columnCount() > 1;

    const auto valueRight = [&](int& outRow, int& outCol) -> std::optional<std::string> {
        const int start = col + spanCols;
        for (int offset = 0; offset < std::max(1, schema_.settings.maxValueDistance); ++offset) {
            const table::TableCell* target = tableValue.cellAt(row, start + offset);
            if (target == nullptr || target->isEmpty()) {
                continue;
            }
            outRow = target->row;
            outCol = target->col;
            return target->text;
        }
        return std::nullopt;
    };
    const auto valueBelow = [&](int& outRow, int& outCol) -> std::optional<std::string> {
        const int start = row + spanRows;
        for (int offset = 0; offset < std::max(1, schema_.settings.maxValueDistance); ++offset) {
            const table::TableCell* target = tableValue.cellAt(start + offset, col);
            if (target == nullptr || target->isEmpty()) {
                continue;
            }
            outRow = target->row;
            outCol = target->col;
            return target->text;
        }
        return std::nullopt;
    };

    std::vector<Candidate> adjacent;
    std::vector<Candidate> labelLike;
    const std::vector<std::function<std::optional<std::string>(int&, int&)>> lookups =
        inHeader ? std::vector<std::function<std::optional<std::string>(int&, int&)>>{valueBelow,
                                                                                     valueRight}
                 : std::vector<std::function<std::optional<std::string>(int&, int&)>>{valueRight,
                                                                                     valueBelow};

    for (const auto& lookup : lookups) {
        int valueRow = 0;
        int valueCol = 0;
        const auto found = lookup(valueRow, valueCol);
        if (!found.has_value() || found->empty()) {
            continue;
        }
        Candidate candidate;
        candidate.source = kSourceTable;
        candidate.rawValue = *found;
        candidate.alias = bestAlias;
        candidate.cellCoords = std::make_pair(valueRow, valueCol);
        // Ô kế bên có thể lại là một nhãn khác (bảng nhãn - giá trị xếp dọc);
        // những ô như vậy bị đẩy xuống cuối danh sách ứng viên.
        if (looksLikeLabel(*found)) {
            candidate.score = bestScore * 0.8;
            labelLike.push_back(std::move(candidate));
        } else {
            candidate.score = bestScore;
            adjacent.push_back(std::move(candidate));
        }
    }

    if (inHeader) {
        // Ở hàng header, giá trị gần như luôn nằm ở ô liền kề (dưới/phải); phần
        // dư ngay trong ô header (VD chú thích mã cột "(TK533)") chỉ được thử SAU
        // CÙNG, để tránh nó "thắng" ô giá trị thật.
        for (auto& candidate : adjacent) {
            candidates.push_back(std::move(candidate));
        }
        if (inlineCandidate.has_value()) {
            candidates.push_back(*inlineCandidate);
        }
    } else {
        if (inlineCandidate.has_value()) {
            candidates.push_back(*inlineCandidate);
        }
        for (auto& candidate : adjacent) {
            candidates.push_back(std::move(candidate));
        }
    }
    for (auto& candidate : labelLike) {
        candidates.push_back(std::move(candidate));
    }

    if (candidates.empty()) {
        log::debug(kLogger, "Field " + spec.name + ": khớp nhãn '" + bestAlias +
                                "' nhưng không tìm thấy ô giá trị");
    }
    return candidates;
}

bool CTKMExtractor::looksLikeLabel(const std::string& text) const {
    if (text.empty()) {
        return false;
    }
    for (const auto& spec : schema_.fields) {
        const AliasMatch match =
            matchAlias(text, spec.aliases, schema_.settings.minSimilarity);
        if (match.alias.has_value() && remainderAfterAlias(text, *match.alias).empty()) {
            return true;
        }
    }
    return false;
}

std::optional<CTKMExtractor::Candidate> CTKMExtractor::findInTextLines(
    const FieldSpec& spec, const std::string& rawText) const {
    if (rawText.empty()) {
        return std::nullopt;
    }
    std::vector<std::string> lines;
    for (const auto& line : splitLines(rawText)) {
        const std::string stripped = trim(line);
        if (!stripped.empty()) {
            lines.push_back(stripped);
        }
    }

    // Nhãn của mọi field KHÁC đều là mốc kết thúc giá trị của field này.
    std::vector<std::string> otherAliases;
    for (const auto& other : schema_.fields) {
        if (other.name == spec.name) {
            continue;
        }
        otherAliases.insert(otherAliases.end(), other.aliases.begin(), other.aliases.end());
    }

    std::optional<Candidate> best;
    for (std::size_t index = 0; index < lines.size(); ++index) {
        const AliasMatch match =
            matchAlias(lines[index], spec.aliases, schema_.settings.minSimilarity);
        if (!match.alias.has_value()) {
            continue;
        }
        const std::string tail = remainderAfterAlias(lines[index], *match.alias);
        std::string value = cutAtNextLabel(tail, spec.aliases, otherAliases);
        // Chỉ nhảy sang dòng kế tiếp khi sau nhãn KHÔNG còn gì. Nếu phần dư có
        // nội dung nhưng bị cắt cụt vì gặp nhãn của field khác thì giá trị của
        // field này thực sự vắng mặt - lấy dòng dưới là lấy nhầm nội dung của
        // cột khác.
        if (tail.empty() && index + 1 < lines.size()) {
            // Nhãn chiếm trọn dòng -> giá trị thường nằm ở dòng kế tiếp.
            const AliasMatch nextMatch =
                matchAlias(lines[index + 1], spec.aliases, schema_.settings.minSimilarity);
            if (nextMatch.score < schema_.settings.minSimilarity) {
                value = cutAtNextLabel(lines[index + 1], spec.aliases, otherAliases);
            }
        }
        if (value.empty()) {
            continue;
        }
        if (!best.has_value() || match.score > best->score) {
            Candidate candidate;
            candidate.source = kSourceTextLine;
            candidate.rawValue = value;
            candidate.score = match.score;
            candidate.alias = match.alias;
            best = candidate;
        }
    }
    return best;
}

std::optional<std::string> CTKMExtractor::findByRegex(const FieldSpec& spec,
                                                      const std::string& rawText) const {
    if (rawText.empty() || spec.regex.empty()) {
        return std::nullopt;
    }
    // Pattern được so khớp trên bản text ĐÃ BỎ DẤU; giá trị trả về vẫn cắt từ
    // text gốc còn dấu nhờ bảng ánh xạ vị trí của FoldedText.
    const FoldedText folded = foldAccents(rawText);

    for (const auto& pattern : spec.regex) {
        std::regex expression;
        try {
            expression = std::regex(pattern, std::regex::icase);
        } catch (const std::regex_error& error) {
            log::warn(kLogger, "Field " + spec.name + ": regex '" + pattern +
                                   "' không hợp lệ (" + error.what() + ")");
            continue;
        }

        std::smatch match;
        if (std::regex_search(folded.folded, match, expression)) {
            const std::size_t group = match.size() > 1 ? 1 : 0;
            const std::size_t start =
                static_cast<std::size_t>(match.position(group));
            const std::size_t end = start + static_cast<std::size_t>(match.length(group));
            const std::size_t originalStart = folded.originalOffset(start);
            const std::size_t originalEnd = folded.originalOffset(end);
            const std::string value =
                trim(rawText.substr(originalStart, originalEnd - originalStart));
            if (!value.empty()) {
                return value;
            }
        }
        if (std::regex_search(rawText, match, expression)) {
            const std::size_t group = match.size() > 1 ? 1 : 0;
            const std::string value = trim(match[group].str());
            if (!value.empty()) {
                return value;
            }
        }
    }
    return std::nullopt;
}

std::optional<FieldValue> CTKMExtractor::coerceType(
    const FieldSpec& spec, const std::optional<FieldValue>& value) const {
    if (!value.has_value()) {
        return std::nullopt;
    }
    std::string declared = spec.type.empty() ? "string" : spec.type;
    std::transform(declared.begin(), declared.end(), declared.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    const bool isNumberType =
        declared == "number" || declared == "float" || declared == "int" || declared == "integer";
    const bool isArrayType = declared == "array" || declared == "list";

    if (isNumberType) {
        if (std::holds_alternative<long long>(*value)) {
            return value;
        }
        if (std::holds_alternative<double>(*value)) {
            if (declared == "int" || declared == "integer") {
                return FieldValue{static_cast<long long>(std::get<double>(*value))};
            }
            return value;
        }
        log::warn(kLogger, "Field " + spec.name + ": giá trị không phải số");
        return std::nullopt;
    }
    if (isArrayType) {
        if (std::holds_alternative<std::vector<std::string>>(*value)) {
            return value;
        }
        if (std::holds_alternative<std::string>(*value)) {
            const std::string text = trim(std::get<std::string>(*value));
            if (!text.empty()) {
                return FieldValue{std::vector<std::string>{text}};
            }
        }
        log::warn(kLogger, "Field " + spec.name + ": giá trị không phải danh sách");
        return std::nullopt;
    }
    if (std::holds_alternative<std::string>(*value)) {
        return value;
    }
    if (std::holds_alternative<long long>(*value)) {
        return FieldValue{std::to_string(std::get<long long>(*value))};
    }
    if (std::holds_alternative<double>(*value)) {
        std::ostringstream stream;
        stream << std::get<double>(*value);
        return FieldValue{stream.str()};
    }
    return value;
}

}  // namespace ctkm::extraction
