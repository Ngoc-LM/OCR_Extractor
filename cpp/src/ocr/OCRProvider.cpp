#include "ocr/OCRProvider.hpp"

#include <sstream>

#include "util/Utf8.hpp"

namespace ctkm::ocr {
namespace {

std::string trim(const std::string& text) {
    const char* whitespace = " \t\r\n\f\v";
    const auto begin = text.find_first_not_of(whitespace);
    if (begin == std::string::npos) {
        return std::string();
    }
    const auto end = text.find_last_not_of(whitespace);
    return text.substr(begin, end - begin + 1);
}

}  // namespace

BoundingBox BoundingBox::fromPolygon(const std::vector<Point>& points) {
    if (points.empty()) {
        throw OCRError("polygon rỗng, không dựng được bounding box");
    }
    double minX = points.front().x;
    double maxX = points.front().x;
    double minY = points.front().y;
    double maxY = points.front().y;
    for (const auto& point : points) {
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
    }
    return BoundingBox(minX, minY, maxX, maxY);
}

double BoundingBox::intersectionArea(const BoundingBox& other) const {
    const double dx = std::min(x2_, other.x2_) - std::max(x1_, other.x1_);
    const double dy = std::min(y2_, other.y2_) - std::max(y1_, other.y1_);
    if (dx <= 0.0 || dy <= 0.0) {
        return 0.0;
    }
    return dx * dy;
}

double BoundingBox::overlapRatio(const BoundingBox& other) const {
    if (area() <= 0.0) {
        return 0.0;
    }
    return intersectionArea(other) / area();
}

double BoundingBox::iou(const BoundingBox& other) const {
    const double inter = intersectionArea(other);
    const double unionArea = area() + other.area() - inter;
    if (unionArea <= 0.0) {
        return 0.0;
    }
    return inter / unionArea;
}

bool BoundingBox::containsCenterOf(const BoundingBox& other) const {
    const Point c = other.center();
    return x1_ <= c.x && c.x <= x2_ && y1_ <= c.y && c.y <= y2_;
}

double BoundingBox::verticalOverlapRatio(const BoundingBox& other) const {
    const double dy = std::min(y2_, other.y2_) - std::max(y1_, other.y1_);
    if (dy <= 0.0) {
        return 0.0;
    }
    const double base = std::min(height(), other.height());
    if (base <= 0.0) {
        return 0.0;
    }
    return dy / base;
}

BoundingBox BoundingBox::merge(const BoundingBox& other) const {
    return BoundingBox(std::min(x1_, other.x1_), std::min(y1_, other.y1_),
                       std::max(x2_, other.x2_), std::max(y2_, other.y2_));
}

OCRToken::OCRToken(std::string tokenText, BoundingBox tokenBox, double tokenConfidence)
    : text(trim(tokenText)), box(tokenBox), confidence(tokenConfidence) {}

std::vector<OCRToken> OCRResult::nonEmptyTokens() const {
    std::vector<OCRToken> result;
    result.reserve(tokens.size());
    for (const auto& token : tokens) {
        if (!token.isEmpty()) {
            result.push_back(token);
        }
    }
    return result;
}

std::string OCRResult::rawText() const {
    const auto lines = groupTokensIntoLines(nonEmptyTokens());
    std::string out;
    for (std::size_t lineIndex = 0; lineIndex < lines.size(); ++lineIndex) {
        if (lineIndex > 0) {
            out += '\n';
        }
        for (std::size_t tokenIndex = 0; tokenIndex < lines[lineIndex].size(); ++tokenIndex) {
            if (tokenIndex > 0) {
                out += ' ';
            }
            out += lines[lineIndex][tokenIndex].text;
        }
    }
    return out;
}

double OCRResult::meanConfidence() const {
    const auto usable = nonEmptyTokens();
    if (usable.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (const auto& token : usable) {
        total += token.confidence;
    }
    return total / static_cast<double>(usable.size());
}

OCRResult OCRResult::fromText(const std::string& text, const std::string& provider) {
    OCRResult result;
    result.provider = provider;
    const double lineHeight = 20.0;
    const double charWidth = 10.0;

    std::istringstream stream(text);
    std::string line;
    int row = 0;
    while (std::getline(stream, line)) {
        const std::string stripped = trim(line);
        if (stripped.empty()) {
            // Dòng rỗng vẫn chiếm 1 chỉ số hàng (bản Python dùng enumerate trên
            // splitlines rồi mới bỏ qua), nhờ vậy toạ độ y của các dòng sau khớp
            // hệt bản Python.
            ++row;
            continue;
        }
        const double y1 = static_cast<double>(row) * (lineHeight + 6.0);
        const double width =
            static_cast<double>(std::max<std::size_t>(1, utf8::length(stripped))) * charWidth;
        result.tokens.emplace_back(stripped, BoundingBox(0.0, y1, width, y1 + lineHeight), 1.0);
        ++row;
    }
    return result;
}

std::vector<OCRToken> sortTokensReadingOrder(const std::vector<OCRToken>& tokens) {
    std::vector<OCRToken> sorted = tokens;
    std::stable_sort(sorted.begin(), sorted.end(), [](const OCRToken& a, const OCRToken& b) {
        if (a.box.y1() != b.box.y1()) {
            return a.box.y1() < b.box.y1();
        }
        return a.box.x1() < b.box.x1();
    });
    return sorted;
}

std::vector<std::vector<OCRToken>> groupTokensIntoLines(const std::vector<OCRToken>& tokens,
                                                        double overlapThreshold) {
    std::vector<std::vector<OCRToken>> lines;
    for (const auto& token : sortTokensReadingOrder(tokens)) {
        bool placed = false;
        for (auto& line : lines) {
            // So với token cuối cùng của dòng: đủ nhanh và ổn định với bảng.
            const OCRToken& reference = line.back();
            if (reference.box.verticalOverlapRatio(token.box) >= overlapThreshold) {
                line.push_back(token);
                placed = true;
                break;
            }
        }
        if (!placed) {
            lines.push_back({token});
        }
    }
    for (auto& line : lines) {
        std::stable_sort(line.begin(), line.end(), [](const OCRToken& a, const OCRToken& b) {
            return a.box.x1() < b.box.x1();
        });
    }
    std::stable_sort(lines.begin(), lines.end(),
                     [](const std::vector<OCRToken>& a, const std::vector<OCRToken>& b) {
                         double minA = a.front().box.y1();
                         for (const auto& token : a) {
                             minA = std::min(minA, token.box.y1());
                         }
                         double minB = b.front().box.y1();
                         for (const auto& token : b) {
                             minB = std::min(minB, token.box.y1());
                         }
                         return minA < minB;
                     });
    return lines;
}

}  // namespace ctkm::ocr
