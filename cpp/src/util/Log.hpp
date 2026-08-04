// Logger tối giản, thay cho module ``logging`` của bản Python.
//
// Mọi log đi ra stderr theo đúng format bản Python: "<LEVEL> <name>: <message>",
// nhờ vậy so sánh log 2 bản khi debug rất nhanh. Mức mặc định là INFO, bật
// ``--debug`` thì chuyển sang DEBUG.
#pragma once

#include <string>

namespace ctkm::log {

enum class Level { Debug = 10, Info = 20, Warning = 30, Error = 40 };

/// Đặt ngưỡng log toàn cục (mặc định ``Level::Info``).
void setLevel(Level level);

/// Ngưỡng log hiện tại.
Level level();

/// Ghi một dòng log nếu ``level`` đạt ngưỡng.
void write(Level level, const std::string& logger, const std::string& message);

inline void debug(const std::string& logger, const std::string& message) {
    write(Level::Debug, logger, message);
}
inline void info(const std::string& logger, const std::string& message) {
    write(Level::Info, logger, message);
}
inline void warn(const std::string& logger, const std::string& message) {
    write(Level::Warning, logger, message);
}
inline void error(const std::string& logger, const std::string& message) {
    write(Level::Error, logger, message);
}

}  // namespace ctkm::log
