#include "util/Utf8.hpp"

namespace ctkm::utf8 {

std::size_t charLength(const std::string& text, std::size_t offset) {
    if (offset >= text.size()) {
        return 0;
    }
    const auto lead = static_cast<unsigned char>(text[offset]);
    std::size_t length = 1;
    if ((lead & 0x80) == 0x00) {
        length = 1;
    } else if ((lead & 0xE0) == 0xC0) {
        length = 2;
    } else if ((lead & 0xF0) == 0xE0) {
        length = 3;
    } else if ((lead & 0xF8) == 0xF0) {
        length = 4;
    } else {
        return 1;  // byte hỏng: coi như 1 ký tự để không kẹt vòng lặp
    }
    if (offset + length > text.size()) {
        return 1;
    }
    for (std::size_t index = 1; index < length; ++index) {
        if (!isContinuation(static_cast<unsigned char>(text[offset + index]))) {
            return 1;
        }
    }
    return length;
}

std::size_t length(const std::string& text) {
    std::size_t count = 0;
    std::size_t offset = 0;
    while (offset < text.size()) {
        offset += charLength(text, offset);
        ++count;
    }
    return count;
}

std::vector<std::string> toCharacters(const std::string& text) {
    std::vector<std::string> characters;
    std::size_t offset = 0;
    while (offset < text.size()) {
        const std::size_t size = charLength(text, offset);
        characters.push_back(text.substr(offset, size));
        offset += size;
    }
    return characters;
}

std::string substr(const std::string& text, std::size_t start, std::size_t count) {
    std::string result;
    std::size_t offset = 0;
    std::size_t index = 0;
    while (offset < text.size() && index < start) {
        offset += charLength(text, offset);
        ++index;
    }
    std::size_t taken = 0;
    while (offset < text.size() && taken < count) {
        const std::size_t size = charLength(text, offset);
        result.append(text, offset, size);
        offset += size;
        ++taken;
    }
    return result;
}

std::string encode(unsigned int codePoint) {
    std::string out;
    if (codePoint < 0x80) {
        out.push_back(static_cast<char>(codePoint));
    } else if (codePoint < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (codePoint >> 6)));
        out.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    } else if (codePoint < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (codePoint >> 12)));
        out.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (codePoint >> 18)));
        out.push_back(static_cast<char>(0x80 | ((codePoint >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    }
    return out;
}

unsigned int decode(const std::string& text, std::size_t& offset) {
    if (offset >= text.size()) {
        return 0;
    }
    const std::size_t size = charLength(text, offset);
    const auto lead = static_cast<unsigned char>(text[offset]);
    unsigned int codePoint = 0;
    switch (size) {
        case 1:
            codePoint = lead;
            break;
        case 2:
            codePoint = lead & 0x1FU;
            break;
        case 3:
            codePoint = lead & 0x0FU;
            break;
        default:
            codePoint = lead & 0x07U;
            break;
    }
    for (std::size_t index = 1; index < size; ++index) {
        codePoint = (codePoint << 6) |
                    (static_cast<unsigned char>(text[offset + index]) & 0x3FU);
    }
    offset += size;
    return codePoint;
}

}  // namespace ctkm::utf8
