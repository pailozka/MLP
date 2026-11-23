#include "mlp/trainer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace mlp {
namespace {
void compatible(const Network& model, const Dataset& data, bool labeled = true) {
    data.validate(labeled);
    if (model.widths().front() != data.features || model.widths().back() != data.classes)
        throw std::invalid_argument("Model and dataset dimensions differ");
}
void check_output(std::ofstream& out, const std::filesystem::path& path) {
    out.flush();
    if (!out) throw std::runtime_error("Cannot write output: " + path.string());
}
void write_indices(const std::filesystem::path& path, const std::vector<std::size_t>& indices) {
    std::ofstream out(path);
    for (auto index : indices) out << index << '\n';
    check_output(out, path);
}
}
Metrics evaluate(const Network& model, const Dataset& data, const std::vector<std::size_t>& indices) {
    compatible(model, data);
    if (indices.empty()) throw std::invalid_argument("Cannot evaluate an empty set");
    Metrics m;
    m.samples = indices.size();
    m.confusion.assign(data.classes, std::vector<std::size_t>(data.classes, 0));
    std::size_t correct = 0;
    for (auto index : indices) {
        const auto p = model.probabilities(data.row(index));
        const auto predicted = static_cast<std::size_t>(std::max_element(p.begin(), p.end()) - p.begin());
        const auto label = data.labels.at(index);
        m.loss += model.loss(data.row(index), label);
        correct += predicted == label;
        ++m.confusion[label][predicted];
    }
    m.loss /= m.samples;
    m.accuracy = static_cast<double>(correct) / m.samples;
    for (std::size_t c = 0; c < data.classes; ++c) {
        std::size_t actual = 0, predicted = 0;
        for (std::size_t j = 0; j < data.classes; ++j) {
            actual += m.confusion[c][j]; predicted += m.confusion[j][c];
        }
        const double tp = static_cast<double>(m.confusion[c][c]);
        m.precision.push_back(predicted ? tp / predicted : 0);
        m.recall.push_back(actual ? tp / actual : 0);
        m.f1.push_back(actual + predicted ? 2 * tp / (actual + predicted) : 0);
        m.macro_f1 += m.f1.back() / data.classes;
    }
    return m;
}
void write_metrics(const Metrics& m, const std::filesystem::path& path) {
    std::ofstream out(path);
    out << std::setprecision(10) << "{\n  \"samples\": " << m.samples
        << ",\n  \"loss\": " << m.loss << ",\n  \"accuracy\": " << m.accuracy
        << ",\n  \"macro_f1\": " << m.macro_f1 << ",\n  \"per_class\": [\n";
    for (std::size_t c = 0; c < m.f1.size(); ++c) {
        out << "    {\"class\": " << c << ", \"precision\": " << m.precision[c]
            << ", \"recall\": " << m.recall[c] << ", \"f1\": " << m.f1[c] << "}"
            << (c + 1 == m.f1.size() ? "\n" : ",\n");
    }
    out << "  ],\n  \"confusion_matrix\": [\n";
    for (std::size_t c = 0; c < m.confusion.size(); ++c) {
        out << "    [";
        for (std::size_t j = 0; j < m.confusion[c].size(); ++j)
            out << (j ? ", " : "") << m.confusion[c][j];
        out << "]" << (c + 1 == m.confusion.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    check_output(out, path);
}
void write_predictions(const Network& model, const Dataset& data, const std::filesystem::path& path) {
    compatible(model, data, false);
    std::ofstream out(path);
    for (std::size_t i = 0; i < data.size(); ++i) {
        auto p = model.probabilities(data.row(i));
        out << std::max_element(p.begin(), p.end()) - p.begin() << '\n';
    }
    check_output(out, path);
}
TrainResult train(Network& model, const Dataset& data, const TrainConfig& c,
                  const std::filesystem::path& output) {
    compatible(model, data);
    if (c.epochs == 0 || c.batch_size == 0 || c.patience == 0)
        throw std::invalid_argument("Epochs, batch size and patience must be positive");
    // Validate optimizer before creating any run artifacts or starting training.
    Network optimizer_probe({1, 2});
    optimizer_probe.step(1, c.optimizer);
    auto split = stratified_split(data, c.validation_fraction, c.seed);
    if (std::filesystem::exists(output) && !std::filesystem::is_empty(output))
        throw std::runtime_error("Run directory must be new or empty: " + output.string());
    std::filesystem::create_directories(output);
    write_indices(output / "train_indices.csv", split.train);
    write_indices(output / "validation_indices.csv", split.validation);
    std::ofstream config(output / "config.json");
    config << std::setprecision(17) << "{\n  \"schema_version\": 1,\n  \"seed\": " << c.seed
           << ",\n  \"widths\": [";
    for (std::size_t i = 0; i < model.widths().size(); ++i) config << (i ? ", " : "") << model.widths()[i];
    config << "],\n  \"parameters\": " << model.parameter_count()
           << ",\n  \"dropout\": " << model.dropout() << ",\n  \"epochs\": " << c.epochs
           << ",\n  \"batch_size\": " << c.batch_size << ",\n  \"patience\": " << c.patience
           << ",\n  \"validation_fraction\": " << c.validation_fraction
           << ",\n  \"training_samples\": " << split.train.size()
           << ",\n  \"validation_samples\": " << split.validation.size()
           << ",\n  \"optimizer\": \"AdamW\",\n  \"learning_rate\": " << c.optimizer.learning_rate
           << ",\n  \"weight_decay\": " << c.optimizer.weight_decay
           << ",\n  \"beta1\": " << c.optimizer.beta1 << ",\n  \"beta2\": " << c.optimizer.beta2
           << ",\n  \"epsilon\": " << c.optimizer.epsilon
           << ",\n  \"clip_norm\": " << c.optimizer.clip_norm
           << ",\n  \"normalization\": \"pixel / 255\"\n}\n";
    check_output(config, output / "config.json");
    std::ofstream history(output / "history.csv");
    history << "epoch,train_loss,validation_loss,validation_accuracy,validation_macro_f1,seconds\n";
    std::mt19937 shuffle_rng(c.seed);
    double best_loss = std::numeric_limits<double>::infinity();
    std::size_t stale = 0;
    std::vector<double> best_parameters;
    TrainResult result;
    for (std::size_t epoch = 1; epoch <= c.epochs; ++epoch) {
        const auto start = std::chrono::steady_clock::now();
        std::shuffle(split.train.begin(), split.train.end(), shuffle_rng);
        double train_loss = 0;
        for (std::size_t begin = 0; begin < split.train.size(); begin += c.batch_size) {
            const auto end = std::min(begin + c.batch_size, split.train.size());
            model.zero_grad();
            for (auto i = begin; i < end; ++i)
                train_loss += model.accumulate_gradient(data.row(split.train[i]), data.labels[split.train[i]]);
            model.step(end - begin, c.optimizer); // Includes the final partial batch.
        }
        const auto metrics = evaluate(model, data, split.validation);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        train_loss /= split.train.size();
        history << std::setprecision(10) << epoch << ',' << train_loss << ',' << metrics.loss << ','
                << metrics.accuracy << ',' << metrics.macro_f1 << ',' << seconds << '\n';
        check_output(history, output / "history.csv");
        std::cout << "Epoch " << epoch << '/' << c.epochs << " train_loss=" << train_loss
                  << " val_loss=" << metrics.loss << " val_accuracy=" << metrics.accuracy
                  << " seconds=" << seconds << std::endl;
        result.epochs_completed = epoch;
        if (metrics.loss < best_loss) {
            best_loss = metrics.loss;
            best_parameters = model.parameters();
            result.best_epoch = epoch;
            result.validation = metrics;
            model.save(output / "best.mlp");
            stale = 0;
        } else if (++stale >= c.patience) break;
    }
    model.set_parameters(best_parameters);
    write_metrics(result.validation, output / "validation_metrics.json");
    std::ofstream summary(output / "summary.json");
    summary << "{\n  \"best_epoch\": " << result.best_epoch << ",\n  \"epochs_completed\": "
            << result.epochs_completed << ",\n  \"selection_metric\": \"validation_loss\"\n}\n";
    check_output(summary, output / "summary.json");
    return result;
}
} // namespace mlp
