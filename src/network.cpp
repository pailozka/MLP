#include "mlp/network.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace mlp {
namespace {
void check_logits(const std::vector<double>& logits) {
    if (logits.empty() || !std::all_of(logits.begin(), logits.end(),
                                     [](double x) { return std::isfinite(x); }))
        throw std::runtime_error("Empty or non-finite logits; check data and learning rate");
}
}

std::vector<double> softmax(const std::vector<double>& logits) {
    check_logits(logits);
    const double maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<double> result(logits.size());
    double total = 0;
    for (std::size_t i = 0; i < logits.size(); ++i)
        total += result[i] = std::exp(logits[i] - maximum);
    for (auto& value : result) value /= total;
    return result;
}

double cross_entropy(const std::vector<double>& logits, std::size_t label) {
    check_logits(logits);
    if (label >= logits.size()) throw std::invalid_argument("Label outside output range");
    const double maximum = *std::max_element(logits.begin(), logits.end());
    double total = 0;
    for (double value : logits) total += std::exp(value - maximum);
    return (maximum - logits[label]) + std::log(total);
}

Network::Network(std::vector<std::size_t> widths, double dropout, std::uint32_t seed)
    : widths_(std::move(widths)), dropout_(dropout), rng_(seed) {
    if (widths_.size() < 2 || widths_.size() > 16 ||
        !std::isfinite(dropout_) || dropout_ < 0 || dropout_ >= 1)
        throw std::invalid_argument("Expected 2..16 layer widths and dropout in [0, 1)");
    std::size_t count = 0;
    for (auto width : widths_)
        if (width == 0 || width > 16384) throw std::invalid_argument("Invalid layer width");
    for (std::size_t l = 1; l < widths_.size(); ++l) {
        count += (widths_[l - 1] + 1) * widths_[l];
        if (count > 10000000) throw std::invalid_argument("Model exceeds 10M parameter limit");
    }
    for (std::size_t l = 1; l < widths_.size(); ++l) {
        Layer layer;
        layer.in = widths_[l - 1];
        layer.out = widths_[l];
        const auto n = layer.in * layer.out;
        layer.weights.resize(n);
        layer.biases.assign(layer.out, 0);
        layer.dw.assign(n, 0); layer.db.assign(layer.out, 0);
        layer.mw.assign(n, 0); layer.vw.assign(n, 0);
        layer.mb.assign(layer.out, 0); layer.vb.assign(layer.out, 0);
        // He normal for ReLU; Xavier normal for the linear output layer.
        const double variance = l + 1 < widths_.size() ? 2.0 / layer.in
                                                     : 2.0 / (layer.in + layer.out);
        std::normal_distribution<double> init(0, std::sqrt(variance));
        for (auto& weight : layer.weights) weight = init(rng_);
        layers_.push_back(std::move(layer));
    }
}

Network::Cache Network::forward(const float* input, bool training, std::mt19937* rng) const {
    if (!input) throw std::invalid_argument("Null model input");
    Cache cache;
    cache.activations.emplace_back(input, input + widths_.front());
    cache.gates.resize(layers_.size());
    std::bernoulli_distribution keep(1.0 - dropout_);
    for (std::size_t l = 0; l < layers_.size(); ++l) {
        const auto& layer = layers_[l];
        const auto& previous = cache.activations.back();
        std::vector<double> current(layer.out);
        const bool hidden = l + 1 < layers_.size();
        if (hidden) cache.gates[l].resize(layer.out);
        for (std::size_t o = 0; o < layer.out; ++o) {
            const double* row = layer.weights.data() + o * layer.in;
            double value = layer.biases[o];
            for (std::size_t i = 0; i < layer.in; ++i) value += row[i] * previous[i];
            if (hidden) {
                double gate = value > 0 ? 1.0 : 0.0;
                if (training && dropout_ > 0) gate *= keep(*rng) ? 1.0 / (1.0 - dropout_) : 0.0;
                cache.gates[l][o] = gate;
                value = gate == 0 ? 0 : value * gate;
            }
            current[o] = value;
        }
        cache.activations.push_back(std::move(current));
    }
    return cache;
}

std::vector<double> Network::probabilities(const float* input) const {
    return softmax(forward(input, false, nullptr).activations.back());
}

double Network::loss(const float* input, std::size_t label) const {
    return cross_entropy(forward(input, false, nullptr).activations.back(), label);
}

void Network::zero_grad() {
    for (auto& layer : layers_) {
        std::fill(layer.dw.begin(), layer.dw.end(), 0);
        std::fill(layer.db.begin(), layer.db.end(), 0);
    }
}

double Network::accumulate_gradient(const float* input, std::size_t label, bool training) {
    auto cache = forward(input, training, &rng_);
    const auto& logits = cache.activations.back();
    const double result = cross_entropy(logits, label);
    auto delta = softmax(logits);
    delta[label] -= 1;
    for (std::size_t l = layers_.size(); l-- > 0;) {
        auto& layer = layers_[l];
        const auto& previous = cache.activations[l];
        std::vector<double> upstream(l > 0 ? layer.in : 0, 0);
        for (std::size_t o = 0; o < layer.out; ++o) {
            layer.db[o] += delta[o];
            double* gradient = layer.dw.data() + o * layer.in;
            const double* weights = layer.weights.data() + o * layer.in;
            for (std::size_t i = 0; i < layer.in; ++i) gradient[i] += delta[o] * previous[i];
            if (l > 0)
                for (std::size_t i = 0; i < layer.in; ++i) upstream[i] += weights[i] * delta[o];
        }
        if (l > 0)
            for (std::size_t i = 0; i < layer.in; ++i) upstream[i] *= cache.gates[l - 1][i];
        delta = std::move(upstream);
    }
    return result;
}

void Network::step(std::size_t batch_size, const AdamWConfig& c) {
    if (batch_size == 0 || !std::isfinite(c.learning_rate) || c.learning_rate <= 0 ||
        !std::isfinite(c.weight_decay) || c.weight_decay < 0 ||
        !std::isfinite(c.beta1) || c.beta1 < 0 || c.beta1 >= 1 ||
        !std::isfinite(c.beta2) || c.beta2 < 0 || c.beta2 >= 1 ||
        !std::isfinite(c.epsilon) || c.epsilon <= 0 ||
        !std::isfinite(c.clip_norm) || c.clip_norm < 0 || c.learning_rate * c.weight_decay >= 1)
        throw std::invalid_argument("Invalid AdamW configuration or batch size");
    double norm = 0;
    for (const auto& layer : layers_) {
        for (double g : layer.dw) norm = std::hypot(norm, g / batch_size);
        for (double g : layer.db) norm = std::hypot(norm, g / batch_size);
    }
    if (!std::isfinite(norm)) throw std::runtime_error("Non-finite gradient");
    double scale = 1.0 / batch_size;
    if (c.clip_norm > 0 && norm > c.clip_norm) scale *= c.clip_norm / norm;
    ++steps_;
    const double correction1 = 1.0 - std::pow(c.beta1, static_cast<double>(steps_));
    const double correction2 = 1.0 - std::pow(c.beta2, static_cast<double>(steps_));
    auto update = [&](std::vector<double>& values, const std::vector<double>& grads,
                      std::vector<double>& m, std::vector<double>& v, double decay) {
        for (std::size_t i = 0; i < values.size(); ++i) {
            const double g = grads[i] * scale;
            m[i] = c.beta1 * m[i] + (1.0 - c.beta1) * g;
            v[i] = c.beta2 * v[i] + (1.0 - c.beta2) * g * g;
            values[i] *= 1.0 - c.learning_rate * decay;
            values[i] -= c.learning_rate * (m[i] / correction1) /
                         (std::sqrt(v[i] / correction2) + c.epsilon);
            if (!std::isfinite(values[i])) throw std::runtime_error("Non-finite model parameter");
        }
    };
    for (auto& layer : layers_) {
        update(layer.weights, layer.dw, layer.mw, layer.vw, c.weight_decay);
        update(layer.biases, layer.db, layer.mb, layer.vb, 0); // Never decay biases.
    }
}

std::size_t Network::parameter_count() const {
    std::size_t count = 0;
    for (const auto& layer : layers_) count += layer.weights.size() + layer.biases.size();
    return count;
}

std::vector<double> Network::parameters() const {
    std::vector<double> result;
    result.reserve(parameter_count());
    for (const auto& layer : layers_) {
        result.insert(result.end(), layer.weights.begin(), layer.weights.end());
        result.insert(result.end(), layer.biases.begin(), layer.biases.end());
    }
    return result;
}
std::vector<double> Network::gradients() const {
    std::vector<double> result;
    result.reserve(parameter_count());
    for (const auto& layer : layers_) {
        result.insert(result.end(), layer.dw.begin(), layer.dw.end());
        result.insert(result.end(), layer.db.begin(), layer.db.end());
    }
    return result;
}
void Network::set_parameters(const std::vector<double>& values) {
    if (values.size() != parameter_count() ||
        !std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("Invalid parameter count or non-finite values");
    auto it = values.begin();
    for (auto& layer : layers_) {
        std::copy_n(it, layer.weights.size(), layer.weights.begin()); it += layer.weights.size();
        std::copy_n(it, layer.biases.size(), layer.biases.begin()); it += layer.biases.size();
        std::fill(layer.mw.begin(), layer.mw.end(), 0); std::fill(layer.vw.begin(), layer.vw.end(), 0);
        std::fill(layer.mb.begin(), layer.mb.end(), 0); std::fill(layer.vb.begin(), layer.vb.end(), 0);
    }
    steps_ = 0;
    zero_grad();
}
void Network::save(const std::filesystem::path& path) const {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write checkpoint: " + path.string());
    out << "MLP_CHECKPOINT 1\n" << widths_.size() << '\n';
    for (auto width : widths_) out << width << ' ';
    out << '\n' << std::setprecision(std::numeric_limits<double>::max_digits10)
        << dropout_ << '\n' << parameter_count() << '\n';
    for (double value : parameters()) out << value << '\n';
    out.flush();
    if (!out) throw std::runtime_error("Failed writing checkpoint: " + path.string());
}
Network Network::load(const std::filesystem::path& path) {
    std::ifstream in(path);
    std::string magic;
    unsigned version = 0;
    std::size_t depth = 0;
    if (!(in >> magic >> version >> depth) || magic != "MLP_CHECKPOINT" || version != 1 ||
        depth < 2 || depth > 16) throw std::runtime_error("Invalid or unsupported checkpoint header");
    std::vector<std::size_t> widths(depth);
    for (auto& width : widths)
        if (!(in >> width)) throw std::runtime_error("Truncated checkpoint architecture");
    double dropout;
    std::size_t count;
    if (!(in >> dropout >> count)) throw std::runtime_error("Truncated checkpoint metadata");
    Network model(widths, dropout);
    if (count != model.parameter_count()) throw std::runtime_error("Checkpoint parameter count mismatch");
    std::vector<double> values(count);
    for (auto& value : values)
        if (!(in >> value)) throw std::runtime_error("Truncated or invalid checkpoint parameters");
    if (in >> magic) throw std::runtime_error("Unexpected trailing checkpoint data");
    model.set_parameters(values);
    return model;
}
} // namespace mlp
