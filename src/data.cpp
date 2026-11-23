#include "mlp/data.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mlp {
namespace {
int parse_integer(std::string_view token, const std::string& context) {
    const auto start = token.find_first_not_of(" \t\r");
    if (start == std::string_view::npos) throw std::runtime_error(context + ": empty field");
    const auto end = token.find_last_not_of(" \t\r");
    token = token.substr(start, end - start + 1);
    int value = 0;
    auto result = std::from_chars(token.data(), token.data() + token.size(), value);
    if (result.ec != std::errc() || result.ptr != token.data() + token.size())
        throw std::runtime_error(context + ": expected integer, found '" + std::string(token) + "'");
    return value;
}
}
const float* Dataset::row(std::size_t index) const {
    if (index >= size()) throw std::out_of_range("Dataset row out of range");
    return inputs.data() + index * features;
}
void Dataset::validate(bool require_labels) const {
    if (features == 0 || classes < 2 || inputs.empty() || inputs.size() % features != 0)
        throw std::invalid_argument("Dataset is empty or has invalid dimensions");
    if ((require_labels || !labels.empty()) && labels.size() != size())
        throw std::invalid_argument("Vector and label row counts differ");
    for (float value : inputs)
        if (!std::isfinite(value) || value < 0 || value > 1)
            throw std::invalid_argument("Input must be finite and normalized to [0,1]");
    for (auto label : labels)
        if (label >= classes) throw std::invalid_argument("Label outside class range");
}
Dataset load_csv(const std::filesystem::path& vectors, const std::filesystem::path& labels,
                 std::size_t features, std::size_t classes) {
    if (features == 0 || classes < 2) throw std::invalid_argument("Invalid dataset dimensions");
    Dataset data{features, classes, {}, {}};
    std::ifstream input(vectors);
    if (!input) throw std::runtime_error("Cannot open vectors: " + vectors.string());
    std::string line;
    std::size_t row = 0;
    while (std::getline(input, line)) {
        ++row;
        if (row == 1 && line.rfind("version https://git-lfs.github.com/", 0) == 0)
            throw std::runtime_error("Training data is a Git LFS pointer. Run: python3 scripts/prepare_data.py");
        const std::string context = vectors.string() + ":" + std::to_string(row);
        std::size_t start = 0, columns = 0;
        do {
            auto end = line.find(',', start);
            if (end == std::string::npos) end = line.size();
            const int value = parse_integer(std::string_view(line).substr(start, end - start), context);
            if (value < 0 || value > 255) throw std::runtime_error(context + ": pixel outside [0,255]");
            if (++columns > features) throw std::runtime_error(context + ": too many columns");
            data.inputs.push_back(static_cast<float>(value) / 255.0f);
            if (end == line.size()) break;
            start = end + 1;
        } while (true);
        if (columns != features) throw std::runtime_error(context + ": wrong feature count");
    }
    if (input.bad()) throw std::runtime_error("Failed reading vectors: " + vectors.string());
    if (!labels.empty()) {
        std::ifstream target(labels);
        if (!target) throw std::runtime_error("Cannot open labels: " + labels.string());
        row = 0;
        while (std::getline(target, line)) {
            const auto context = labels.string() + ":" + std::to_string(++row);
            const int label = parse_integer(line, context);
            if (label < 0 || static_cast<std::size_t>(label) >= classes)
                throw std::runtime_error(context + ": label outside class range");
            data.labels.push_back(static_cast<std::size_t>(label));
        }
        if (target.bad()) throw std::runtime_error("Failed reading labels: " + labels.string());
    }
    data.validate(!labels.empty());
    return data;
}
Split stratified_split(const Dataset& data, double fraction, std::uint32_t seed) {
    data.validate();
    if (!std::isfinite(fraction) || fraction <= 0 || fraction >= 1)
        throw std::invalid_argument("Validation fraction must be in (0,1)");
    std::vector<std::vector<std::size_t>> by_class(data.classes);
    for (std::size_t i = 0; i < data.size(); ++i) by_class[data.labels[i]].push_back(i);
    std::mt19937 rng(seed);
    Split result;
    for (auto& group : by_class) {
        if (group.size() < 2)
            throw std::invalid_argument("Stratified split requires at least two examples per class");
        std::shuffle(group.begin(), group.end(), rng);
        const auto count = std::clamp(static_cast<std::size_t>(std::llround(group.size() * fraction)),
                                      std::size_t{1}, group.size() - 1);
        result.validation.insert(result.validation.end(), group.begin(), group.begin() + count);
        result.train.insert(result.train.end(), group.begin() + count, group.end());
    }
    std::shuffle(result.train.begin(), result.train.end(), rng);
    std::shuffle(result.validation.begin(), result.validation.end(), rng);
    return result;
}
std::vector<std::size_t> all_indices(const Dataset& data) {
    std::vector<std::size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
}
} // namespace mlp
