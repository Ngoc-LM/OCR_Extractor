// Tiện ích UTF-8 dùng chung.
//
// Bản Python thao tác trên chuỗi Unicode nên len()/slicing tính theo KÝ TỰ.
// Bản C++ dùng std::string (byte), vì vậy mọi chỗ bản Python đếm ký tự
// (max_length của text_parser, min_item_length của list_parser, căn lề bảng khi
// in debug) đều phải dùng các hàm ở đây để cho ra cùng kết quả.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ctkm::utf8 {

/// Số ký tự (code point) của một chuỗi UTF-8; byte lỗi được tính là 1 ký tự.
std::size_t length(const std::string& text);

/// Cắt chuỗi theo chỉ số KÝ TỰ (không phải byte), giống ``text[start:start+count]``.
std::string substr(const std::string& text, std::size_t start, std::size_t count);

/// Tách chuỗi thành danh sách các ký tự UTF-8 (mỗi phần tử là 1 code point).
std::vector<std::string> toCharacters(const std::string& text);

/// True nếu byte là byte tiếp diễn của một ký tự nhiều byte (10xxxxxx).
inline bool isContinuation(unsigned char byte) { return (byte & 0xC0) == 0x80; }

/// Độ dài (byte) của ký tự UTF-8 bắt đầu tại ``offset``.
///
/// Trả 0 khi ``offset`` đã ra ngoài chuỗi, còn lại luôn >= 1 (byte hỏng vẫn được
/// tính là 1 để vòng lặp duyệt chuỗi không bị kẹt).
std::size_t charLength(const std::string& text, std::size_t offset);

/// Chuyển code point sang chuỗi UTF-8.
std::string encode(unsigned int codePoint);

/// Giải mã ký tự UTF-8 tại ``offset``; trả về code point, cập nhật ``offset``.
unsigned int decode(const std::string& text, std::size_t& offset);

}  // namespace ctkm::utf8
