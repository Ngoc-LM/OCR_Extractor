#include "util/Log.hpp"

#include <iostream>

namespace ctkm::log {
namespace {

Level g_level = Level::Info;

const char* levelName(Level level) {
    switch (level) {
        case Level::Debug:
            return "DEBUG";
        case Level::Info:
            return "INFO";
        case Level::Warning:
            return "WARNING";
        case Level::Error:
            return "ERROR";
    }
    return "INFO";
}

}  // namespace

void setLevel(Level level) { g_level = level; }

Level level() { return g_level; }

void write(Level level, const std::string& logger, const std::string& message) {
    if (static_cast<int>(level) < static_cast<int>(g_level)) {
        return;
    }
    std::cerr << levelName(level) << ' ' << logger << ": " << message << '\n';
}

}  // namespace ctkm::log
