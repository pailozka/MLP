#include "mlp/trainer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void close(double actual, double expected, double tolerance, const std::string& message) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
            message + ": actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
}
void throws(const std::function<void()>& action, const std::string& message) {
    try { action(); } catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}
void write(const fs::path& path, const std::string& content) { std::ofstream(path) << content; }

void stable_math() {
    auto p = mlp::softmax({10000, 10001, 9999});
    close(std::accumulate(p.begin(), p.end(), 0.0), 1, 1e-14, "softmax sums to one");
    auto shifted = mlp::softmax({0, 1, -1});
    for (std::size_t i = 0; i < p.size(); ++i) close(p[i], shifted[i], 1e-14, "shift invariance");
    close(mlp::cross_entropy({1000, -1000}, 1), 2000, 1e-12, "CE must not clip saturated probabilities");
    throws([] { mlp::softmax({}); }, "empty logits accepted");
    throws([] { mlp::cross_entropy({1, 2}, 2); }, "invalid label accepted");
    throws([] { mlp::Network bad({0, 10}); }, "zero width accepted");
    throws([] { mlp::Network bad({2, 2}, 1.0); }, "invalid dropout accepted");
}

void gradient_check(double dropout) {
    mlp::Network model({3, 4, 3, 2}, dropout, 123);
    const float input[] = {0.2f, 0.7f, 0.9f};
    const auto initial = model.parameters();
    auto analytic_model = model;
    analytic_model.accumulate_gradient(input, 1, true);
    const auto analytic = analytic_model.gradients();
    double largest = 0;
    for (std::size_t i = 0; i < initial.size(); ++i) {
        constexpr double epsilon = 1e-5;
        auto plus = model, minus = model;
        auto values = initial;
        values[i] += epsilon; plus.set_parameters(values);
        values[i] -= 2 * epsilon; minus.set_parameters(values);
        // Copies start from the same RNG state, holding the dropout mask fixed.
        const double numerical = (plus.accumulate_gradient(input, 1) - minus.accumulate_gradient(input, 1)) / (2 * epsilon);
        const double error = std::abs(numerical - analytic[i]);
        largest = std::max(largest, error);
        close(analytic[i], numerical, 1e-7, "finite difference parameter " + std::to_string(i));
    }
    auto a = model.probabilities(input);
    model.accumulate_gradient(input, 0);
    require(a == model.probabilities(input), "dropout affected inference");
    std::cout << "  gradient max absolute error (dropout=" << dropout << "): " << largest << '\n';
}

void adam_and_batches() {
    mlp::Network model({2, 2}, 0, 4);
    const float input[] = {0.3f, 0.8f};
    const auto before = model.parameters();
    model.accumulate_gradient(input, 1);
    auto gradient = model.gradients();
    auto duplicate = model;
    duplicate.accumulate_gradient(input, 1);
    mlp::AdamWConfig config;
    config.learning_rate = 0.01; config.weight_decay = 0.1; config.clip_norm = 0;
    model.step(1, config); duplicate.step(2, config);
    require(model.parameters() == duplicate.parameters(), "batch gradients not averaged");
    auto after = model.parameters();
    std::vector<double> first_m(gradient.size()), first_v(gradient.size());
    for (std::size_t i = 0; i < after.size(); ++i) {
        const double decay = i < 4 ? 0.1 : 0;
        const double expected = before[i] * (1 - 0.01 * decay) - 0.01 * gradient[i] / (std::abs(gradient[i]) + config.epsilon);
        close(after[i], expected, 1e-12, "AdamW first step, bias decay exclusion");
        first_m[i] = 0.1 * gradient[i]; first_v[i] = 0.001 * gradient[i] * gradient[i];
    }
    model.zero_grad(); model.accumulate_gradient(input, 0);
    gradient = model.gradients();
    model.step(1, config);
    auto second = model.parameters();
    for (std::size_t i = 0; i < second.size(); ++i) {
        const double m = (0.9 * first_m[i] + 0.1 * gradient[i]) / (1 - 0.9 * 0.9);
        const double v = (0.999 * first_v[i] + 0.001 * gradient[i] * gradient[i]) / (1 - 0.999 * 0.999);
        const double decay = i < 4 ? 0.1 : 0;
        close(second[i], after[i] * (1 - 0.01 * decay) - 0.01 * m / (std::sqrt(v) + config.epsilon),
              1e-12, "persistent moments and bias correction");
    }
    throws([&] { model.step(0, config); }, "empty batch accepted");
    model.zero_grad();
    for (double value : model.gradients()) close(value, 0, 0, "gradient reset");
}

mlp::Dataset toy_data() {
    mlp::Dataset data{2, 2, {}, {}};
    for (int repeat = 0; repeat < 4; ++repeat) {
        data.inputs.insert(data.inputs.end(), {0, 0, 0, 1, 1, 0, 1, 1});
        data.labels.insert(data.labels.end(), {0, 1, 1, 0});
    }
    return data;
}
void learn_nonlinear() {
    auto data = toy_data();
    mlp::Network model({2, 12, 2}, 0, 7);
    mlp::AdamWConfig config; config.learning_rate = 0.03; config.weight_decay = 0;
    auto initial = mlp::evaluate(model, data, mlp::all_indices(data));
    for (int step = 0; step < 500; ++step) {
        model.zero_grad();
        for (std::size_t i = 0; i < data.size(); ++i) model.accumulate_gradient(data.row(i), data.labels[i]);
        model.step(data.size(), config);
    }
    auto final = mlp::evaluate(model, data, mlp::all_indices(data));
    require(final.loss < initial.loss * 0.05 && final.accuracy == 1, "network failed to learn XOR");
    close(final.macro_f1, 1, 0, "perfect F1");
}
void data_and_checkpoint(const fs::path& root) {
    const auto vectors = root / "vectors.csv", labels = root / "labels.csv";
    write(vectors, "0,255\r\n 128,64 \n"); write(labels, "0\n1\n");
    auto data = mlp::load_csv(vectors, labels, 2, 2);
    close(data.row(0)[1], 1, 0, "pixel normalization");
    close(data.row(1)[0], 128.0 / 255, 1e-7, "pixel normalization");
    require(mlp::load_csv(vectors, {}, 2, 2).labels.empty(), "unlabeled inference");
    for (const auto& invalid : {"1\n", "1,2,3\n", "1,\n", "1,256\n", "-1,2\n", "1,nan\n", "1,1.2\n", "\n", ""}) {
        write(vectors, invalid);
        throws([&] { mlp::load_csv(vectors, {}, 2, 2); }, "invalid CSV accepted");
    }
    write(vectors, "version https://git-lfs.github.com/spec/v1\n");
    throws([&] { mlp::load_csv(vectors, {}, 2, 2); }, "LFS pointer accepted");
    write(vectors, "1,2\n");
    throws([&] { mlp::load_csv(vectors, labels, 2, 2); }, "label count mismatch accepted");
    write(labels, "2\n");
    throws([&] { mlp::load_csv(vectors, labels, 2, 2); }, "out of range label accepted");
    mlp::Network model({2, 3, 2}, 0.25, 17);
    auto path = root / "model.mlp";
    model.save(path);
    auto loaded = mlp::Network::load(path);
    require(model.parameters() == loaded.parameters() && loaded.dropout() == 0.25, "checkpoint round trip");
    require(model.probabilities(data.row(0)) == loaded.probabilities(data.row(0)), "checkpoint changed prediction");
    write(path, "MLP_CHECKPOINT 99\n2\n");
    throws([&] { mlp::Network::load(path); }, "unsupported version accepted");
    write(path, "MLP_CHECKPOINT 1\n2\n2 2\n0\n6\n0\n");
    throws([&] { mlp::Network::load(path); }, "truncated checkpoint accepted");
}
void split_and_trainer(const fs::path& root) {
    auto data = toy_data();
    const auto split = mlp::stratified_split(data, 0.25, 42);
    const auto again = mlp::stratified_split(data, 0.25, 42);
    require(split.train == again.train && split.validation == again.validation, "split is not repeatable");
    require(split.train.size() == 12 && split.validation.size() == 4, "wrong split sizes");
    std::set<std::size_t> indices(split.train.begin(), split.train.end());
    for (auto index : split.validation) require(indices.insert(index).second, "validation leakage");
    require(indices.size() == data.size(), "split lost samples");
    for (std::size_t c = 0; c < 2; ++c) {
        std::size_t count = 0;
        for (auto index : split.validation) count += data.labels[index] == c;
        require(count == 2, "split not stratified");
    }
    mlp::Network model({2, 2}, 0, 42), manual = model;
    mlp::TrainConfig config;
    config.epochs = 1; config.batch_size = 5; config.validation_fraction = 0.25;
    auto order = split.train;
    std::mt19937 rng(config.seed); std::shuffle(order.begin(), order.end(), rng);
    for (std::size_t begin = 0; begin < order.size(); begin += 5) {
        const auto end = std::min(begin + 5, order.size());
        manual.zero_grad();
        for (auto i = begin; i < end; ++i) manual.accumulate_gradient(data.row(order[i]), data.labels[order[i]]);
        manual.step(end - begin, config.optimizer);
    }
    const auto result = mlp::train(model, data, config, root / "run");
    require(model.parameters() == manual.parameters(), "trainer dropped or mis-scaled partial batch");
    require(result.best_epoch == 1, "wrong best epoch");
    throws([&] { mlp::train(model, data, config, root / "run"); }, "existing run overwritten");
    // Updates below floating-point resolution ensure an exact validation plateau.
    config.epochs = 10; config.patience = 2; config.optimizer.learning_rate = 1e-30;
    auto result2 = mlp::train(model, data, config, root / "plateau");
    require(result2.best_epoch == 1 && result2.epochs_completed == 3, "patience off by one");
    require(model.parameters() == mlp::Network::load(root / "plateau/best.mlp").parameters(), "best weights not restored");
}
int main() {
    const auto root = fs::temp_directory_path() /
        ("mlp-tests-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    try {
        stable_math();
        gradient_check(0); gradient_check(0.35);
        adam_and_batches(); learn_nonlinear(); data_and_checkpoint(root); split_and_trainer(root);
        fs::remove_all(root);
        std::cout << "All numerical, data, checkpoint and training checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        fs::remove_all(root);
        std::cerr << "FAIL: " << error.what() << '\n'; return 1;
    }
}
