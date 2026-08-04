#include "ocr/OnnxVietOCR.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>

#include "util/Log.hpp"

#ifdef CTKM_WITH_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace ctkm::ocr {
namespace {

constexpr const char* kLogger = "ctkm.ocr.vietocr";

/// Resize giữ tỉ lệ về chiều cao chuẩn rồi chia 255 -> [0, 1], layout NCHW.
[[maybe_unused]] std::vector<float> prepareCrop(const cv::Mat& crop,
                                                const RecognizerConfig& config,
                                                int& outWidth) {
    cv::Mat bgr = crop;
    if (bgr.channels() == 1) {
        cv::cvtColor(crop, bgr, cv::COLOR_GRAY2BGR);
    }
    // Bám sát ``vietocr.tool.translate.resize``: lấy phần nguyên của tỉ lệ rồi
    // LÀM TRÒN LÊN bội số của 10 trước khi kẹp vào [minWidth, maxWidth]. Model
    // được huấn luyện với đúng các bề rộng này.
    const double ratio = static_cast<double>(bgr.cols) / std::max(1, bgr.rows);
    int width = static_cast<int>(config.imageHeight * ratio);
    width = static_cast<int>(std::ceil(width / 10.0)) * 10;
    width = std::clamp(width, config.minWidth, config.maxWidth);

    cv::Mat resized;
    // vietocr dùng ``Image.LANCZOS``; INTER_LANCZOS4 là bộ lọc gần nhất của OpenCV.
    cv::resize(bgr, resized, cv::Size(width, config.imageHeight), 0, 0, cv::INTER_LANCZOS4);
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    cv::Mat asFloat;
    // ``vietocr.tool.translate.process_image`` chỉ chia 255 -> giá trị nằm trong
    // [0, 1]. KHÔNG chuẩn hoá tiếp về [-1, 1]: model được huấn luyện trên [0, 1]
    // nên lệch dải giá trị là ra text rác.
    rgb.convertTo(asFloat, CV_32FC3, 1.0 / 255.0);

    const int height = asFloat.rows;
    std::vector<float> tensor(static_cast<std::size_t>(3) * height * width);
    for (int y = 0; y < height; ++y) {
        const auto* row = asFloat.ptr<cv::Vec3f>(y);
        for (int x = 0; x < width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                tensor[(static_cast<std::size_t>(channel) * height + y) * width + x] =
                    row[x][channel];
            }
        }
    }
    outWidth = width;
    return tensor;
}

/// Softmax ổn định số học trên một vector logits.
[[maybe_unused]] std::vector<float> softmax(const float* logits, std::size_t size) {
    std::vector<float> probabilities(size);
    if (size == 0) {
        return probabilities;
    }
    const float maximum = *std::max_element(logits, logits + size);
    float total = 0.0F;
    for (std::size_t index = 0; index < size; ++index) {
        probabilities[index] = std::exp(logits[index] - maximum);
        total += probabilities[index];
    }
    if (total > 0.0F) {
        for (auto& value : probabilities) {
            value /= total;
        }
    }
    return probabilities;
}

}  // namespace

std::vector<std::string> loadVocabulary(const std::string& vocabPath) {
    std::ifstream stream(vocabPath, std::ios::binary);
    if (!stream) {
        throw ProviderUnavailableError("Không đọc được bảng ký tự VietOCR: " + vocabPath);
    }
    std::vector<std::string> vocabulary;
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        vocabulary.push_back(line);
    }
    if (vocabulary.size() < 5) {
        throw ProviderUnavailableError("Bảng ký tự VietOCR quá ngắn: " + vocabPath);
    }
    return vocabulary;
}

#ifdef CTKM_WITH_ONNXRUNTIME

struct OnnxVietOCR::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ctkm-vietocr"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
};

bool OnnxVietOCR::isSupported() { return true; }

OnnxVietOCR::OnnxVietOCR(const std::string& modelPath, const std::string& vocabPath,
                         const RecognizerConfig& config)
    : impl_(std::make_unique<Impl>()), config_(config) {
    if (!std::filesystem::is_regular_file(modelPath)) {
        throw ProviderUnavailableError("Không tìm thấy model VietOCR: " + modelPath);
    }
    std::string resolvedVocab = vocabPath;
    if (resolvedVocab.empty()) {
        const std::filesystem::path model(modelPath);
        const std::filesystem::path sibling = model.parent_path() / "vietocr_vocab.txt";
        const std::filesystem::path suffixed = std::filesystem::path(modelPath + ".vocab");
        resolvedVocab = std::filesystem::is_regular_file(suffixed) ? suffixed.string()
                                                                  : sibling.string();
    }
    vocabulary_ = loadVocabulary(resolvedVocab);

    try {
        impl_->options.SetIntraOpNumThreads(1);
        impl_->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        impl_->session =
            std::make_unique<Ort::Session>(impl_->env, modelPath.c_str(), impl_->options);
        const std::size_t inputCount = impl_->session->GetInputCount();
        for (std::size_t index = 0; index < inputCount; ++index) {
            impl_->inputNames.emplace_back(
                impl_->session->GetInputNameAllocated(index, impl_->allocator).get());
        }
        const std::size_t outputCount = impl_->session->GetOutputCount();
        for (std::size_t index = 0; index < outputCount; ++index) {
            impl_->outputNames.emplace_back(
                impl_->session->GetOutputNameAllocated(index, impl_->allocator).get());
        }
    } catch (const std::exception& error) {
        throw ProviderUnavailableError(std::string("Không nạp được model VietOCR: ") +
                                       error.what());
    }
}

OnnxVietOCR::~OnnxVietOCR() = default;
OnnxVietOCR::OnnxVietOCR(OnnxVietOCR&&) noexcept = default;
OnnxVietOCR& OnnxVietOCR::operator=(OnnxVietOCR&&) noexcept = default;

RecognitionResult OnnxVietOCR::recognize(const cv::Mat& crop) {
    RecognitionResult result;
    if (crop.empty()) {
        return result;
    }

    int width = 0;
    std::vector<float> imageTensor;
    try {
        imageTensor = prepareCrop(crop, config_, width);
    } catch (const std::exception& error) {
        log::warn(kLogger, std::string("Chuẩn bị crop thất bại: ") + error.what());
        return result;
    }

    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 4> imageShape{1, 3, config_.imageHeight, width};

    std::vector<int64_t> generated{config_.sosId};
    std::string text;
    double confidenceSum = 0.0;
    int confidenceCount = 0;

    const bool autoRegressive = impl_->inputNames.size() >= 2;
    const int steps = autoRegressive ? config_.maxLength : 1;

    for (int step = 0; step < steps; ++step) {
        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(memoryInfo, imageTensor.data(),
                                                         imageTensor.size(), imageShape.data(),
                                                         imageShape.size()));
        std::array<int64_t, 2> targetShape{1, static_cast<int64_t>(generated.size())};
        if (autoRegressive) {
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memoryInfo, generated.data(), generated.size(), targetShape.data(),
                targetShape.size()));
        }

        std::vector<const char*> inputNames;
        inputNames.reserve(inputs.size());
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            inputNames.push_back(impl_->inputNames[index].c_str());
        }
        std::vector<const char*> outputNames{impl_->outputNames.front().c_str()};

        std::vector<Ort::Value> outputs;
        try {
            outputs = impl_->session->Run(Ort::RunOptions{nullptr}, inputNames.data(),
                                          inputs.data(), inputs.size(), outputNames.data(),
                                          outputNames.size());
        } catch (const std::exception& error) {
            log::warn(kLogger, std::string("VietOCR nhận dạng thất bại: ") + error.what());
            return result;
        }
        if (outputs.empty()) {
            return result;
        }

        const auto info = outputs.front().GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> shape = info.GetShape();
        if (shape.size() < 2) {
            log::warn(kLogger, "Shape output của VietOCR không hợp lệ");
            return result;
        }
        const std::size_t vocabSize = static_cast<std::size_t>(shape.back());
        const std::size_t timeSteps =
            shape.size() >= 3 ? static_cast<std::size_t>(shape[shape.size() - 2]) : 1;
        const float* data = outputs.front().GetTensorData<float>();

        if (autoRegressive) {
            // Chỉ lấy bước cuối cùng của chuỗi đã sinh.
            const float* lastStep = data + (timeSteps - 1) * vocabSize;
            const std::vector<float> probabilities = softmax(lastStep, vocabSize);
            const auto best = static_cast<int64_t>(
                std::distance(probabilities.begin(),
                              std::max_element(probabilities.begin(), probabilities.end())));
            if (best == config_.eosId || best == config_.padId) {
                break;
            }
            confidenceSum += probabilities[static_cast<std::size_t>(best)];
            ++confidenceCount;
            if (best >= 0 && static_cast<std::size_t>(best) < vocabulary_.size()) {
                text += vocabulary_[static_cast<std::size_t>(best)];
            }
            generated.push_back(best);
        } else {
            // Model xuất trọn chuỗi: duyệt toàn bộ bước thời gian một lần.
            for (std::size_t position = 0; position < timeSteps; ++position) {
                const std::vector<float> probabilities =
                    softmax(data + position * vocabSize, vocabSize);
                const auto best = static_cast<std::size_t>(std::distance(
                    probabilities.begin(),
                    std::max_element(probabilities.begin(), probabilities.end())));
                if (static_cast<int>(best) == config_.eosId) {
                    break;
                }
                if (static_cast<int>(best) == config_.padId ||
                    static_cast<int>(best) == config_.sosId) {
                    continue;
                }
                confidenceSum += probabilities[best];
                ++confidenceCount;
                if (best < vocabulary_.size()) {
                    text += vocabulary_[best];
                }
            }
        }
    }

    result.text = text;
    result.confidence = confidenceCount > 0 ? confidenceSum / confidenceCount : 0.0;
    return result;
}

#else  // CTKM_WITH_ONNXRUNTIME

struct OnnxVietOCR::Impl {};

bool OnnxVietOCR::isSupported() { return false; }

OnnxVietOCR::OnnxVietOCR(const std::string& modelPath, const std::string& vocabPath,
                         const RecognizerConfig& config)
    : impl_(nullptr), config_(config) {
    (void)modelPath;
    (void)vocabPath;
    throw ProviderUnavailableError(
        "Bản build này không có ONNXRuntime - cấu hình lại CMake với "
        "-DONNXRUNTIME_ROOT_DIR=<đường dẫn> để dùng provider paddle_vietocr");
}

OnnxVietOCR::~OnnxVietOCR() = default;
OnnxVietOCR::OnnxVietOCR(OnnxVietOCR&&) noexcept = default;
OnnxVietOCR& OnnxVietOCR::operator=(OnnxVietOCR&&) noexcept = default;

RecognitionResult OnnxVietOCR::recognize(const cv::Mat& crop) {
    (void)crop;
    return RecognitionResult{};
}

#endif  // CTKM_WITH_ONNXRUNTIME

}  // namespace ctkm::ocr
