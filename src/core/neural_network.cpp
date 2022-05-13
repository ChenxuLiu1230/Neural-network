
#include "../../include/core/neural_network.h"

namespace neuralnetwork {

// This constructor mainly does some initialization of vectors.
NeuralNetwork::NeuralNetwork() {

  // Here we reserve the extra 1 slot in each vector for the bias term, although we haven't added this term yet.
  weight_from_input_to_hidden_ = vector<vector<double>> (kInputNeurons + 1, vector<double>(kHiddenNeurons + 1, 0));
  weight_from_hidden_to_output_ = vector<vector<double>> (kHiddenNeurons + 1, vector<double>(kOutputNeurons + 1, 0));
  InitializeWeights();

  gradient_from_input_to_hidden_ = vector<vector<double>> (kInputNeurons + 1, vector<double>(kHiddenNeurons + 1, 0));
  gradient_from_hidden_to_output_ = vector<vector<double>> (kHiddenNeurons + 1, vector<double>(kOutputNeurons + 1, 0));

  input_out_ = vector<double> (kInputNeurons + 1, 0);
  hidden_in_ = vector<double> (kHiddenNeurons + 1, 0);
  hidden_out_ = vector<double> (kHiddenNeurons + 1, 0);
  output_in_ = vector<double> (kOutputNeurons + 1, 0);
  output_out_ = vector<double> (kOutputNeurons + 1, 0);

  gradient_loss_to_output_in_ = vector<double> (kOutputNeurons + 1, 0);
  gradient_loss_to_hidden_in_ = vector<double> (kHiddenNeurons + 1, 0);

  // shaded_info_ = vector<vector<bool>> (kImageSize, vector<bool> (kImageSize, false));
  label_ = vector<double> (kOutputNeurons + 1, 0);

}

void NeuralNetwork::InitializeWeights() {
  // We will follow simple heuristics, initialize all weights between -1 and 1.
  std::default_random_engine generator;
  std::uniform_real_distribution<double> weights_initialization(-1, 1);

  // Initialize weights from input layer to hidden layer.
  for (int i = 1; i <= kInputNeurons; i++) {
    for (int j = 1; j <= kHiddenNeurons; j++) {
      weight_from_input_to_hidden_[i][j] = weights_initialization(generator);
    }
  }

  // Initialize weights from hidden layer to output layer.
  for (int i = 1; i <= kHiddenNeurons; i++) {
    for (int j = 1; j <= kOutputNeurons; j++) {
      weight_from_hidden_to_output_[i][j] = weights_initialization(generator);
    }
  }

}

void NeuralNetwork::LoadOneImage(ifstream & input_file) {
  string line;
  getline(input_file, line);
  stringstream ss(line);
  string token;

  // Parse image label.
  fill(label_.begin(), label_.end(), 0.0);
  getline(ss, token, ',');
  curr_digit_ = stoi(token);
  label_[curr_digit_ + 1] = 1.0;

  // Parse image's shaded information.
  int index = 1;
  while (getline(ss, token, ',')) {
    // cout << stoi(token) << " ";
    if (stoi(token) != 0) {
      input_out_[index] = 1;
    } else {
      input_out_[index] = 0;
    }

    index++;
  }

}

void NeuralNetwork::ForwardPropagation() {
  // Reset all values back to zero beforehand
  fill(hidden_in_.begin(), hidden_in_.end(), 0);
  fill(hidden_out_.begin(), hidden_out_.end(), 0);
  fill(output_in_.begin(), output_in_.end(), 0);
  fill(output_out_.begin(), output_out_.end(), 0);

  // Forward from input layer to hidden layer.
  for (int j = 1; j <= kHiddenNeurons; j++) {
    for (int i = 1; i <= kInputNeurons; i++) {
      hidden_in_[j] += input_out_[i] * weight_from_input_to_hidden_[i][j];
    }
  }

  // Apply activation function on hidden layer.
  for (int j = 1; j <= kHiddenNeurons; j++) {
    hidden_out_[j] = SigmoidActivation(hidden_in_[j]);
  }

  // Forward from hidden layer to output layer.
  for (int j = 1; j <= kOutputNeurons; j++) {
    for (int i = 1; i <= kHiddenNeurons; i++) {
      output_in_[j] += hidden_out_[i] * weight_from_hidden_to_output_[i][j];
    }
  }

  // Apply activation function on output layer.
  for (int j = 1; j <= kOutputNeurons; j++) {
    output_out_[j] = SigmoidActivation(output_in_[j]);
  }

}

double NeuralNetwork::SigmoidActivation(double x) {
  return 1.0 / (1.0 + exp(-x));
}

void NeuralNetwork::BackwardPropagation() {
  fill(gradient_loss_to_output_in_.begin(), gradient_loss_to_output_in_.end(), 0);
  fill(gradient_loss_to_hidden_in_.begin(), gradient_loss_to_hidden_in_.end(), 0);
  gradient_from_hidden_to_output_ = vector<vector<double>> (kHiddenNeurons + 1, vector<double>(kOutputNeurons + 1, 0));
  gradient_from_input_to_hidden_ = vector<vector<double>> (kInputNeurons + 1, vector<double>(kHiddenNeurons + 1, 0));

  // Perform gradient descent for gradients from hidden to output layer.
  for (int j = 1; j <= kOutputNeurons; j++) {
    gradient_loss_to_output_in_[j] = (label_[j] - output_out_[j]) * (-1) * output_out_[j] * (1 - output_out_[j]);
    for (int i = 1; i <= kHiddenNeurons; i++) {
      gradient_from_hidden_to_output_[i][j] = gradient_loss_to_output_in_[j] * hidden_out_[i];
      // Gradient descent.
      weight_from_hidden_to_output_[i][j] -= learning_rate * gradient_from_hidden_to_output_[i][j];
    }
  }

  // Perform gradient descent for gradients from input to hidden layer.
  for (int j = 1; j <= kHiddenNeurons; j++) {
    for (int k = 1; k <= kOutputNeurons; k++) {
      gradient_loss_to_hidden_in_[j] += gradient_loss_to_output_in_[k] * weight_from_hidden_to_output_[j][k];
    }
    gradient_loss_to_hidden_in_[j] *= hidden_out_[j] * (1 - hidden_out_[j]);

    for (int i = 1; i <= kInputNeurons; i++) {
      gradient_from_input_to_hidden_[i][j] = gradient_loss_to_hidden_in_[j] * input_out_[i];
      // Gradient descent.
      weight_from_input_to_hidden_[i][j] -= learning_rate * gradient_from_input_to_hidden_[i][j];
    }
  }

}

double NeuralNetwork::GetLoss() const {
  double loss = 0;
  for (int i = 1; i <= kOutputNeurons; i++) {
    loss += pow(label_[i] - output_out_[i], 2);
  }

  loss *= 0.5;
  return loss;

}

void NeuralNetwork::Train(ifstream & input_file) {
  if (input_file.is_open()) {

    for (int i = 0; i < kNumTrainingImages; i++) {
      LoadOneImage(input_file);
      for (int k = 0; k < epochs; k++) {
        ForwardPropagation();
        if (GetLoss() < threshold) {
          break;
        } else {
          BackwardPropagation();
        }
      }
    }

    input_file.close();
  }
}

int NeuralNetwork::Predict() {
  ForwardPropagation();
  int max_index = 1;  // the index 1 refers to digit '0', index 2 refers to digit '1', etc.
  double max_prob = output_out_[1];
  for (int i = 2; i <= kOutputNeurons; i++) {
    if (output_out_[i] > max_prob) {
      max_prob = output_out_[i];
      max_index = i;
    }
  }

  max_index--;
  return max_index;
}

double NeuralNetwork::Validate(ifstream & input_file) {
  // cout << "validation file open: " << input_file.is_open() << endl;
  if (input_file.is_open()) {
    int count = 0;
    for (int k = 0; k < kNumValidationImages; k++) {
      LoadOneImage(input_file);
      // ForwardPropagation();
      int max_index = Predict();
       cout << "max_index: " << max_index << ", " << "curr_digit_: " << curr_digit_ << endl;
      if (max_index == curr_digit_) {
        count++;
      }
    }

    input_file.close();
    return 1.0 * count / kNumValidationImages;

  } else {
    return 0;
  }

}

void NeuralNetwork::SaveToFile() {
  ofstream model_weights(kModelWeightsFile);
  if (model_weights.is_open()) {
    for (int i = 1; i < kInputNeurons; i++) {
      for (int j = 1; j < kHiddenNeurons; j++) {
        model_weights << weight_from_input_to_hidden_[i][j] << " ";
      }
      model_weights << endl;
    }

    for (int i = 1; i < kHiddenNeurons; i++) {
      for (int j = 1; j < kOutputNeurons; j++) {
        model_weights << weight_from_hidden_to_output_[i][j] << " ";
      }
      model_weights << endl;
    }

    model_weights.close();
  }

}

void NeuralNetwork::LoadFromFile() {
  ifstream model_weights(kModelWeightsFile);
  // cout << "model weights open: " << model_weights.is_open() << endl;
  // cout << model_weights.is_open() << endl;
  if (model_weights.is_open()) {
    for (int i = 1; i <= kInputNeurons; i++) {
      for (int j = 1; j <= kHiddenNeurons; j++) {
        model_weights >> weight_from_input_to_hidden_[i][j];
      }
    }

    for (int i = 1; i <= kHiddenNeurons; i++) {
      for (int j = 1; j <= kOutputNeurons; j++) {
        model_weights >> weight_from_hidden_to_output_[i][j];
      }
    }

    model_weights.close();
  }
}

} // namespace neuralnetwork