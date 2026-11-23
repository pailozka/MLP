#include "mlp/trainer.h"

#include <charconv>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace {
using Options = std::map<std::string, std::string>;
std::string get(const Options& options, const std::string& name, const std::string& fallback = "") {
    const auto it = options.find(name);
    return it == options.end() ? fallback : it->second;
}
std::string required(const Options& options, const std::string& name) {
    auto value = get(options, name);
    if (value.empty()) throw std::invalid_argument("Missing required option " + name);
    return value;
}
std::size_t integer(const std::string& text) {
    std::size_t result = 0;
    auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size())
        throw std::invalid_argument("Invalid unsigned integer: " + text);
    return result;
}
double number(const std::string& text) {
    std::size_t end = 0;
    double value = std::stod(text, &end);
    if (end != text.size() || !std::isfinite(value)) throw std::invalid_argument("Invalid number: " + text);
    return value;
}
void help() {
    std::cout << R"(Fashion-MNIST MLP — C++17, no ML framework

  mlp train [options]
    --data-dir PATH       CSV directory (default: data/prepared)
    --output PATH         New/empty run directory (default: runs/default)
    --hidden 128,64       Comma-separated hidden widths; 'none' for linear baseline
    --dropout 0.1         Hidden dropout probability, [0,1)
    --epochs 20          Maximum epochs
    --batch-size 64      True mini-batch size, including partial final batch
    --lr 0.001           AdamW learning rate
    --weight-decay 0.0001  Decoupled weight decay (weights only)
    --clip-norm 5        Global gradient norm limit; 0 disables clipping
    --val-fraction 0.1   Fixed stratified validation fraction
    --patience 5         Stop after this many epochs without lower validation loss
    --seed 42            Initialization, dropout and split/shuffle seed

  mlp evaluate --model PATH --vectors PATH --labels PATH [--output PATH]
    Writes metrics.json and predictions.csv; default output: runs/evaluation

  mlp predict --model PATH --vectors PATH --output FILE
    Writes one integer class per line; labels are not required.

Training never reads the test set. Checkpoints are for inference, not resuming AdamW.
CSV: 784 integer pixels in [0,255] per row; labels: one integer in [0,9] per row.
)";
}
}
int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
            help(); return 0;
        }
        const std::string command = argv[1];
        std::set<std::string> allowed;
        if (command == "train") allowed = {"--data-dir", "--output", "--hidden", "--dropout", "--epochs",
            "--batch-size", "--lr", "--weight-decay", "--clip-norm", "--val-fraction", "--patience", "--seed"};
        else if (command == "evaluate") allowed = {"--model", "--vectors", "--labels", "--output"};
        else if (command == "predict") allowed = {"--model", "--vectors", "--output"};
        else throw std::invalid_argument("Unknown command: " + command);
        Options options;
        for (int i = 2; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") { help(); return 0; }
            if (!allowed.count(key)) throw std::invalid_argument("Unknown option: " + key);
            if (i + 1 == argc || std::string(argv[i + 1]).rfind("--", 0) == 0)
                throw std::invalid_argument("Missing value for " + key);
            if (!options.emplace(key, argv[++i]).second) throw std::invalid_argument("Duplicate option: " + key);
        }
        if (command == "train") {
            mlp::TrainConfig config;
            config.epochs = integer(get(options, "--epochs", "20"));
            config.batch_size = integer(get(options, "--batch-size", "64"));
            config.patience = integer(get(options, "--patience", "5"));
            const auto seed = integer(get(options, "--seed", "42"));
            if (seed > std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("Seed exceeds uint32 range");
            config.seed = static_cast<std::uint32_t>(seed);
            config.validation_fraction = number(get(options, "--val-fraction", "0.1"));
            config.optimizer.learning_rate = number(get(options, "--lr", "0.001"));
            config.optimizer.weight_decay = number(get(options, "--weight-decay", "0.0001"));
            config.optimizer.clip_norm = number(get(options, "--clip-norm", "5"));
            std::vector<std::size_t> widths{784};
            const auto hidden = get(options, "--hidden", "128,64");
            if (hidden != "none") {
                std::size_t begin = 0;
                while (true) {
                    const auto end = hidden.find(',', begin);
                    widths.push_back(integer(hidden.substr(begin, end == std::string::npos ? end : end - begin)));
                    if (end == std::string::npos) break;
                    begin = end + 1;
                }
            }
            widths.push_back(10);
            mlp::Network model(widths, number(get(options, "--dropout", "0.1")), config.seed);
            const std::filesystem::path data_dir = get(options, "--data-dir", "data/prepared");
            auto data = mlp::load_csv(data_dir / "fashion_mnist_train_vectors.csv",
                                      data_dir / "fashion_mnist_train_labels.csv");
            std::cout << "Loaded " << data.size() << " samples; " << model.parameter_count() << " parameters" << std::endl;
            const auto result = mlp::train(model, data, config, get(options, "--output", "runs/default"));
            std::cout << "Best epoch: " << result.best_epoch << "; validation accuracy: "
                      << result.validation.accuracy << std::endl;
        } else {
            auto model = mlp::Network::load(required(options, "--model"));
            const auto vectors = required(options, "--vectors");
            const auto labels = command == "evaluate" ? required(options, "--labels") : "";
            auto data = mlp::load_csv(vectors, labels, model.widths().front(), model.widths().back());
            const std::filesystem::path output = command == "predict" ? required(options, "--output")
                                                                     : get(options, "--output", "runs/evaluation");
            if (command == "predict") {
                mlp::write_predictions(model, data, output);
            } else {
                if (std::filesystem::exists(output) && !std::filesystem::is_empty(output))
                    throw std::runtime_error("Evaluation directory must be new or empty: " + output.string());
                auto metrics = mlp::evaluate(model, data, mlp::all_indices(data));
                std::filesystem::create_directories(output);
                mlp::write_metrics(metrics, output / "metrics.json");
                mlp::write_predictions(model, data, output / "predictions.csv");
                std::cout << "Accuracy: " << metrics.accuracy << "; macro F1: " << metrics.macro_f1
                          << "; loss: " << metrics.loss << '\n';
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
