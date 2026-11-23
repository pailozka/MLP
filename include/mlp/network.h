#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <random>
#include <vector>

namespace mlp {

struct AdamWConfig {
    double learning_rate = 0.001;
    double weight_decay = 0.0001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    double clip_norm = 5.0; // Zero disables global gradient clipping.
};

// Dense ReLU classifier. The final layer produces logits for softmax CE.
// Owns its parameters and optimizer state; independent models share no globals.
class Network {
public:
    Network(std::vector<std::size_t> widths, double dropout = 0.0,
            std::uint32_t seed = 42);
    const std::vector<std::size_t>& widths() const { return widths_; }
    double dropout() const { return dropout_; }
    std::size_t parameter_count() const;
    std::vector<double> probabilities(const float* input) const;
    double loss(const float* input, std::size_t label) const;

    void zero_grad();
    double accumulate_gradient(const float* input, std::size_t label,
                               bool training = true);
    void step(std::size_t batch_size, const AdamWConfig& config);

    // Flattened access supports independent finite-difference verification.
    std::vector<double> parameters() const;
    std::vector<double> gradients() const;
    void set_parameters(const std::vector<double>& values);

    // Versioned inference checkpoint. Optimizer/RNG state is intentionally absent.
    void save(const std::filesystem::path& path) const;
    static Network load(const std::filesystem::path& path);

private:
    struct Layer {
        std::size_t in, out;
        std::vector<double> weights, biases, dw, db, mw, vw, mb, vb;
    };
    struct Cache {
        std::vector<std::vector<double>> activations;
        std::vector<std::vector<double>> gates;
    };
    Cache forward(const float* input, bool training, std::mt19937* rng) const;
    std::vector<std::size_t> widths_;
    double dropout_;
    std::vector<Layer> layers_;
    std::mt19937 rng_;
    std::size_t steps_ = 0;
};

std::vector<double> softmax(const std::vector<double>& logits);
double cross_entropy(const std::vector<double>& logits, std::size_t label);

} // namespace mlp
