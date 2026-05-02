//
//
// NeuralNetwork_Project.cpp
// KP & RM
//
#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif
#include <fstream>
#include <ctime>
#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <cmath>
#include "activation.h"
#include "data.h"

const int BatchSize = 16;
const float beta1 = 0.9;
const float beta2 = 0.999;

//Input Layer
const int Input_Layer = 784;

//Hidden layer
const int Hidden_Layer1 = 128;
const int Hidden_Layer2 = 128;

//Output Layer
const int Output_Layer = 10;

const int epoch = 60;
const float learning_rate = 0.001;

const float momentum = 0.9;
const float epsilon = 1;
const float stabilization = 1e-8;

float Weights_h1[Hidden_Layer1][Input_Layer];
float Weights_h2[Hidden_Layer2][Hidden_Layer1];
float Weights_o[Output_Layer][Hidden_Layer2];

float Deltas_w_h1[Hidden_Layer1][Input_Layer];
float Deltas_w_h2[Hidden_Layer2][Hidden_Layer1];
float Deltas_w_o[Output_Layer][Hidden_Layer2];

float Bias_h1[Hidden_Layer1];
float Bias_h2[Hidden_Layer2];
float Bias_o[Output_Layer];

float Input[Input_Layer];
float Output_h1[Hidden_Layer1]; //input of hidden_layer2
float Output_h2[Hidden_Layer2]; //input of output
float Output_o[Output_Layer];

float Output_Expected[Output_Layer];

float Potential_h1[Hidden_Layer1];
float Potential_h2[Hidden_Layer2];
float Potential_o[Output_Layer];

float Deltas_h1[Hidden_Layer1];
float Deltas_h2[Hidden_Layer2];
float Deltas_o[Output_Layer];

float Train_Data[Train_Data_size][Input_Layer];
float Train_Output[Train_Data_size][Output_Layer];
float Prediction_Train_Output[Train_Data_size];

float Test_Data[Test_Data_size][Input_Layer];
float Test_Output[Test_Data_size][Output_Layer];
float Prediction_Test_Output[Test_Data_size];

std::vector<int> train_index;
std::vector<int> validation_index;

void Init_Input(int index, bool Training) {
    for (int i = 0; i < Input_Layer; i++) {
        if (Training) {
            Input[i] = Train_Data[index][i];
        }
        else
            Input[i] = Test_Data[index][i];
    }
}

void Init_Output_Expected(int index, bool Training) {
    for (int i = 0; i < Output_Layer; i++) {
        if (Training) {
            Output_Expected[i] = Train_Output[index][i];
        }
        else
            Output_Expected[i] = Test_Output[index][i];
    }
}


void SetPredictionValue(size_t index, bool train) {
    float max = 0.0;
    size_t max_index;

    for (size_t i = 0; i < Output_Layer; i++) {
        if (max < Output_o[i]) {
            max = Output_o[i];
            max_index = i;
        }
    }

    if (train)
        Prediction_Train_Output[index] = max_index;
    else
        Prediction_Test_Output[index] = max_index;
}

void CreatePredictionFile(bool train) {

    char buff[FILENAME_MAX];
    GetCurrentDir( buff, FILENAME_MAX );
    std::string current_working_dir(buff);

    if (train)
        current_working_dir += "/train_predictions.csv";
    else
        current_working_dir += "/test_predictions.csv";

    std::ofstream  predictionfile;
    predictionfile.open(current_working_dir);

    if (train){
        for (size_t i = 0; i < Train_Data_size; i++) {
            predictionfile << Prediction_Train_Output[i];
            if (i + 1 < Train_Data_size)
                predictionfile << "\n";
        }
    }
    else {
        for (size_t i = 0; i < Test_Data_size; i++) {
            predictionfile << Prediction_Test_Output[i];
            if (i + 1 < Test_Data_size)
                predictionfile << "\n";
        }
    }
}

void Init_Weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> DistRelu(0, (float) 2 / Input_Layer);
    std::uniform_real_distribution<> DistSelu(0, (float) 1 / Input_Layer);
    std::uniform_real_distribution<> DistSoft(0, (float) 2 / (Output_Layer + Hidden_Layer1));

    for (size_t i = 0; i < Hidden_Layer1; i++) {
        Bias_h1[i] = DistRelu(gen);
        for (size_t j = 0; j < Input_Layer; j++) {
            Weights_h1[i][j] = DistSoft(gen);
        }
    }

    for (size_t i = 0; i < Hidden_Layer2; i++) {
        Bias_h2[i] = DistRelu(gen);
        for (size_t j = 0; j < Hidden_Layer1; j++) {
            Weights_h2[i][j] = DistSoft(gen);
        }
    }

    for (size_t i = 0; i < Output_Layer; i++) {
        Bias_o[i] = DistSoft(gen);
        for (size_t j = 0; j < Hidden_Layer2; j++) {
            Weights_o[i][j] = DistSoft(gen);
        }
    }
}


void Forward_Propagation() {
    for (size_t i = 0; i < Hidden_Layer1; i++) {
        float tmp_potential = Bias_h1[i];
        for (size_t j = 0; j < Input_Layer; j++) {
            tmp_potential += Weights_h1[i][j] * Input[j];
        }
        Potential_h1[i] = tmp_potential;
        Output_h1[i] = Sigmoid(tmp_potential);
    }

    for (size_t i = 0; i < Hidden_Layer2; i++) {
        float tmp_potential = Bias_h2[i];

        for (size_t j = 0; j < Hidden_Layer1; j++) {
            tmp_potential += Weights_h2[i][j] * Output_h1[j];
        }

        Potential_h2[i] = tmp_potential;
        Output_h2[i] = Sigmoid(tmp_potential);
    }

    // FOR final layer softmax is used (with log-sum-exp for numerical stability)
    float max_pot = -1e30f;
    for (size_t i = 0; i < Output_Layer; i++) {
        float tmp_potential = Bias_o[i];
        for (size_t j = 0; j < Hidden_Layer2; j++) {
            tmp_potential += Weights_o[i][j] * Output_h2[j];
        }
        Potential_o[i] = tmp_potential;
        if (tmp_potential > max_pot) max_pot = tmp_potential;
    }

    float sum = 0.0;
    for (size_t i = 0; i < Output_Layer; i++) {
        sum += exp(Potential_o[i] - max_pot);
    }
    for (size_t i = 0; i < Output_Layer; i++) {
        Output_o[i] = exp(Potential_o[i] - max_pot) / sum;
    }
}

float cross_entropy() {
    float loss = 0.0;
    for (size_t i = 0; i < Output_Layer; i++) {
        loss += -1 * (Output_Expected[i] * std::log(Output_o[i] + stabilization));
    }
    return loss;
}

void accuracy(float& acc) {
    float max = 0.0;
    size_t max_index;

    for (size_t i = 0; i < Output_Layer; i++) {
        if (max < Output_o[i]) {
            max = Output_o[i];
            max_index = i;
        }
    }

    if (Output_Expected[max_index] == 1)
        acc++;
}

void Backward_Propagation() {
    float total_error;
    for (size_t i = 0; i < Output_Layer; i++) {
        Deltas_o[i] = Output_o[i] - Output_Expected[i];
    }

    for (size_t i = 0; i < Hidden_Layer2; i++) {
        total_error = 0.0;

        for (size_t j = 0; j < Output_Layer; j++) {
            total_error += Deltas_o[j] * Weights_o[j][i];
        }

        Deltas_h2[i] = total_error * Derivate_Sigmoid(Output_h2[i]);
    }

    for (size_t i = 0; i < Hidden_Layer1; i++) {
        total_error = 0.0;

        for (size_t j = 0; j < Hidden_Layer2; j++) {
            total_error += Deltas_h2[j] * Weights_h2[j][i];
        }
        Deltas_h1[i] = total_error * Derivate_Sigmoid(Output_h1[i]);
    }


    // Weight update
    for (size_t i = 0; i < Hidden_Layer1; i++) {
        Bias_h1[i] += -learning_rate * Deltas_h1[i];

        for (size_t j = 0; j < Input_Layer; j++) {
            Deltas_w_h1[i][j] = -learning_rate * Deltas_h1[i] * Input[j] + Deltas_w_h1[i][j] * momentum;
            Weights_h1[i][j] += Deltas_w_h1[i][j];

        }
    }

    for (size_t i = 0; i < Hidden_Layer2; i++) {
        Bias_h2[i] += -learning_rate * Deltas_h2[i];

        for (size_t j = 0; j < Hidden_Layer1; j++) {
            Deltas_w_h2[i][j] = -learning_rate * Deltas_h2[i] * Output_h1[j] + Deltas_w_h2[i][j] * momentum;
            Weights_h2[i][j] += Deltas_w_h2[i][j];
        }
    }

    for (size_t i = 0; i < Output_Layer; i++) {
        Bias_o[i] += -learning_rate * Deltas_o[i];
        for (size_t j = 0; j < Hidden_Layer2; j++) {
            Deltas_w_o[i][j] = -learning_rate * Deltas_o[i] * Output_h2[j] + Deltas_w_o[i][j] * momentum;
            Weights_o[i][j] += Deltas_w_o[i][j];
        }
    }

}

void Reinit_moments() {
    for (size_t i = 0; i < Hidden_Layer1; i++) {
        for (size_t j = 0; j < Input_Layer; j++) {
            Deltas_w_h1[i][j] = 0.0;
        }
    }

    for (size_t i = 0; i < Hidden_Layer2; i++) {
        for (size_t j = 0; j < Hidden_Layer1; j++) {
            Deltas_w_h2[i][j] = 0;
        }
    }

    for (size_t i = 0; i < Output_Layer; i++) {
        for (size_t j = 0; j < Hidden_Layer2; j++) {
            Deltas_w_o[i][j] = 0;
        }
    }
}

int learning(int tmp_index) {
    Init_Input(tmp_index, true);
    Init_Output_Expected(tmp_index, true);
    Forward_Propagation();
    Backward_Propagation();
    return 0;
}


void shuffle_data(int &seed) {
    srand(seed);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::unordered_set<int> set_random;
    std::uniform_int_distribution<int> dist{ 0, Train_Data_size - 1 };
    while (set_random.size() != Train_Data_size) {
        set_random.insert((int) rand() % (Train_Data_size));
    }

    std::unordered_set<int> set_random_validation;
    std::uniform_int_distribution<int> dist_validation{ 0, 6000 - 1 };
    while (set_random_validation.size() != 6000) {
        int val_index = (int) rand() % (Train_Data_size);
        auto it = set_random.find(val_index);
        if (it != set_random.end()) {
            set_random.erase(it);
        }
        set_random_validation.insert(val_index);
    }
    train_index.clear();
    validation_index.clear();
    train_index.insert(train_index.end(), set_random.begin(), set_random.end());
    validation_index.insert(validation_index.end(), set_random_validation.begin(), set_random_validation.end());
    seed++;
}

float validate() {
    float acc = 0.0;
    int data_size = 0;

    for (auto j : validation_index) {
        Init_Input(j, true);
        Init_Output_Expected(j, true);
        Forward_Propagation();
        accuracy(acc);
    }
    return acc;
}


int Train() {

    int seed = 1;
    int i = 0;

    for (; i < epoch; i++) {
        shuffle_data(seed);
        Reinit_moments();
        int index = 0;
        int train_samples = (int)(Train_Data_size * 0.9);
        train_samples -= train_samples % BatchSize;

        for (int j = 0; j < train_samples; j += BatchSize) {
            for (int b = 0; b < BatchSize; b++) {
                learning(train_index[index]);
                index++;
            }
        }

        float acc = validate();
        float val_acc = acc / validation_index.size();
        printf("Epoch %d val_acc: %.4f\n", i + 1, val_acc);
        if (val_acc > 0.94) {
            return i;
        }
    }
    return i;
}

void Testing(bool train) {
    float acc = 0.0;
    int data_size = 0;
    if (train)
        data_size = Train_Data_size;
    else
        data_size = Test_Data_size;
    for (int j = 0; j < data_size; j++) {
        Init_Input(j, train);
        Init_Output_Expected(j, train);
        Forward_Propagation();
        accuracy(acc);
        SetPredictionValue(j, train);

    }
    if (train)
        printf("Accuracy %% : %f  \n\n", acc * 100 / Train_Data_size);
    else
        printf("Accuracy %% : %f  \n\n", acc * 100 / Test_Data_size);
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    printf("starting to get the data\n");
    Get_Input_Data(Train_Data, true);
    Get_Input_Data(Test_Data, false);
    Get_Output_Data(Train_Output, true);
    Get_Output_Data(Test_Output, false);
    Init_Weights();
    printf("starting to train\n");
    int epochs = Train();
    printf("Epochs: %d\n", epochs);
    Testing(true);
    CreatePredictionFile(true);
    Testing(false);
    CreatePredictionFile(false);
    printf("Batch: %d\n Epoch:  %d\n LR: %f\n H1: %d\n H2: %d\n", BatchSize, epoch, learning_rate, Hidden_Layer1, Hidden_Layer2);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(stop - start);

    std::cout << "Time taken by function: "
              << duration.count() << " Minutes" << std::endl;
}