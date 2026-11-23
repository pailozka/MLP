#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace mlp {
struct Dataset {
    std::size_t features = 0;
    std::size_t classes = 0;
    std::vector<float> inputs;
    std::vector<std::size_t> labels;
    std::size_t size() const { return features ? inputs.size() / features : 0; }
    const float* row(std::size_t index) const;
    void validate(bool require_labels = true) const;
};

// Strict headerless integer CSV, pixels in [0,255], normalized to [0,1].
Dataset load_csv(const std::filesystem::path& vectors,
                 const std::filesystem::path& labels = {},
                 std::size_t features = 784, std::size_t classes = 10);
struct Split { std::vector<std::size_t> train, validation; };
Split stratified_split(const Dataset& data, double validation_fraction, std::uint32_t seed);
std::vector<std::size_t> all_indices(const Dataset& data);
} // namespace mlp
