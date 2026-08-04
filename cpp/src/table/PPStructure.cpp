#include "table/PPStructure.hpp"

#include <utility>

namespace ctkm::table {

PPStructureTableRecognizer::PPStructureTableRecognizer(std::string modelPath)
    : modelPath_(std::move(modelPath)) {}

bool PPStructureTableRecognizer::isAvailable() { return false; }

Table PPStructureTableRecognizer::recognize(const cv::Mat& image,
                                            const std::vector<OCRToken>& tokens) const {
    (void)image;
    (void)tokens;
    throw TableStructureUnavailableError(
        "PP-Structure chưa được port sang C++ (stretch goal) - dùng morphology hoặc "
        "cluster bounding box");
}

Table PPStructureTableRecognizer::recognize(const std::string& imagePath,
                                            const std::vector<OCRToken>& tokens) const {
    (void)imagePath;
    (void)tokens;
    throw TableStructureUnavailableError(
        "PP-Structure chưa được port sang C++ (stretch goal) - dùng morphology hoặc "
        "cluster bounding box");
}

}  // namespace ctkm::table
