

#pragma once

#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>


using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::stringstream;
using std::cout;
using std::endl;

namespace neuralnetwork {

class NeuralNetwork {

public:

 /**
  * Default constructor for NeuralNetwork class, initialize member vectors.
  */
  NeuralNetwork();

  /**
   * Helper function to initialize weights in this deep nets.
   */
  void InitializeWeights();

  /**
   * Helper function to parse one input image from the file.
   * @param input_file the input file stream of the file to be read.
   */
  void LoadOneImage(ifstream & input_file);

  /**
   * Forward propagation of the deep nets to compute model loss.
   */
  void ForwardPropagation();

  /**
   * Sigmoid function for activation of output layer.
   * @param x the input x value
   * @return the y value after applying sigmoid function.
   */
  double SigmoidActivation(double x);

  /**
   * Backward propagation of the deep nets to perform gradient descent for each weight in the network.
   */
  void BackwardPropagation();

  /**
   * Compute the loss (prediction error) of the neural network based on the current weights.
   * @return the loss of the model.
   */
  double GetLoss() const;

  /**
   * Train model.
   * @param input_file the input filestream of training dataset.
   */
  void Train(ifstream & input_file);

  /**
   * Prediction of class digit given new input image.
   * @return the digit number with the largest probability.
   */
  int Predict();

  /**
   * Cross-validation.
   * @param input_file the input filestream of validating dataset.
   * @return
   */
  double Validate(ifstream & input_file);

  /**
   * Save model weights into file to avoid repeated training, which is extremely time-costly.
   * The file path is specified by kModelWeightsFile.
   */
  void SaveToFile();

  /**
   * Load model weights from file.
   * The path is specified by kModelWeightsFile.
   */
  void LoadFromFile();

//private:

  const string kTrainingDataFile = "../data/MNIST_train.txt";
  const string kValidationDataFile = "../data/MNIST_test.txt";
  const string kModelWeightsFile = "/Users/479846747qq.com/Desktop/Cinder/cinder_0.9.2_mac/my-projects/final-project-ChenxuLiu1230/data/model_weights.txt";
  const int kImageSize = 28;
  const int kInputNeurons = 784;
  const int kHiddenNeurons = 128;
  const int kOutputNeurons = 10;
  const int epochs = 1000;
  const double learning_rate = 0.001;
  const double threshold = 0.005;

  // The MNIST contains 60000 images in the training dataset and 10000 in the validation/test set.
  const int kNumTrainingImages = 60000;
  const int kNumValidationImages = 10000;

  vector<vector<double>> weight_from_input_to_hidden_;

  vector<vector<double>> weight_from_hidden_to_output_;

  vector<vector<double>> gradient_from_input_to_hidden_;

  vector<vector<double>> gradient_from_hidden_to_output_;

  // This is the vector storing the input image's shaded information in row-wise direction (the value of input layer neurons)
  vector<double> input_out_;

  vector<double> hidden_in_;

  vector<double> hidden_out_;

  vector<double> output_in_;

  // This is the final predicted probability for each class digit.
  vector<double> output_out_;

  // Theta3, gradient of loss to the input of output layer.
  vector<double> gradient_loss_to_output_in_;

  // Theta2, gradient of loss to the input of hidden layer.
  vector<double> gradient_loss_to_hidden_in_;

  // One-hot vector that represents image's label
  vector<double> label_;

  int curr_digit_;
};

} // namespace neuralnetwork



