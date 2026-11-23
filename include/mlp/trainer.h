#pragma once

#include "mlp/data.h"
#include "mlp/network.h"
#include <filesystem>
#include <vector>

namespace mlp {
struct Metrics {
    std::size_t samples = 0;
    double loss = 0, accuracy = 0, macro_f1 = 0;
    std::vector<std::vector<std::size_t>> confusion;
    std::vector<double> precision, recall, f1;
};
Metrics evaluate(const Network& model, const Dataset& data,
                 const std::vector<std::size_t>& indices);
void write_metrics(const Metrics& metrics, const std::filesystem::path& path);
void write_predictions(const Network& model, const Dataset& data,
                       const std::filesystem::path& path);

struct TrainConfig {
    std::size_t epochs = 20, batch_size = 64, patience = 5;
    double validation_fraction = 0.1;
    std::uint32_t seed = 42;
    AdamWConfig optimizer;
};
struct TrainResult {
    std::size_t best_epoch = 0, epochs_completed = 0;
    Metrics validation;
};
// Creates a fixed split once, shuffles training indices only, restores best weights.
TrainResult train(Network& model, const Dataset& data, const TrainConfig& config,
                  const std::filesystem::path& output);
} // namespace mlp
