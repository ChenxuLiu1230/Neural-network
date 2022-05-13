#pragma once

#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <iostream>
using std::vector;
using std::ifstream;
using std::ofstream;
using std::endl;


namespace autoencoder {

class AutoEncoder {
 public:

  /**
   * Constructor for AutoEncoder class.
   */
  AutoEncoder();

  /**
   * Forward data from input layer, through hidden layer to output layer.
   */
  void ForwardPropagate();

  /**
   * Backward propagation, perform gradient descents for each weight.
   */
  void BackwardPropagate();

  /**
   * Helper function to apply sigmoid activation.
   * @param x the input value
   * @return the value after applying sigmoid activation.
   */
  double SigmoidActivation(double x);

  /**
   * Get training loss of the model.
   * @return the loss of the model. (squared loss between input and ouput vectors)
   */
  double GetLoss();

  /**
   * Initialize weights for the two weights matrix -- from input to latent, from latent to output.
   */
  void InitializeWeights();

  /**
   * Load one input image for a generative model (no label any more).
   * @param input_file the filestream for the input data.
   */
  void LoadOneImage(ifstream & input_file);

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

  /**
   * Train model.
   * @param input_file the input filestream of training dataset.
   */
  void Train(ifstream & input_file);



 private:
  const int kInputNeurons = 784;   // This is also the dimension for output data.
  const int kLatentNeurons = 196;  // Try compressing original image into 14 * 14 images.
  const int kNumTrainingImages = 60000;
  const int epochs = 500;
  const double learning_rate_ = 0.005;
  const double error_threshold = 0.005;
  const string kModelWeightsFile = "";     // TODO: Add path of model weights file.
  vector<double> input_layer_;
  vector<double> latent_layer_;
  vector<double> output_layer_;
  vector<vector<double>> weights_from_input_to_latent_;
  vector<vector<double>> weights_from_latent_to_output_;
  vector<vector<double>> gradient_from_input_to_latent_;
  vector<vector<double>> gradient_from_latent_to_output_;
  vector<double> gradient_;
  vector<double> gradient_of_loss_;
};

} // namespace autoencoder.
