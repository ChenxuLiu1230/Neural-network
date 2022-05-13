#include <string>
#include <sstream>
using std::string;
using std::stringstream;

#include "../../include/core/autoencoder.h"

namespace autoencoder {

AutoEncoder::AutoEncoder() {
  input_layer_ = vector<double> (kInputNeurons, 0);
  latent_layer_ = vector<double> (kLatentNeurons, 0);
  output_layer_ = vector<double> (kInputNeurons, 0);
  weights_from_input_to_latent_ = vector<vector<double>> (kInputNeurons, vector<double>(kLatentNeurons, 0));
  weights_from_latent_to_output_ = vector<vector<double>> (kLatentNeurons, vector<double>(kInputNeurons, 0));
  InitializeWeights();
  gradient_from_input_to_latent_ = vector<vector<double>> (kInputNeurons, vector<double>(kLatentNeurons, 0));
  gradient_from_latent_to_output_ = vector<vector<double>> (kLatentNeurons, vector<double>(kInputNeurons, 0));
  gradient_ = vector<double> (kInputNeurons, 0);
  gradient_of_loss_ = vector<double> (kLatentNeurons, 0);
}

void AutoEncoder::ForwardPropagate() {
  // From input layer to latent layer (encoder).
  for (int j = 0; j < kLatentNeurons; j++) {
    double sum = 0;
    for (int i = 0; i < kInputNeurons; i++) {
      sum += input_layer_[i] * weights_from_input_to_latent_[i][j];
    }
    latent_layer_[j] = SigmoidActivation(sum);
  }

  // From latent layer to output layer (decoder).
  for (int j = 0; j < kInputNeurons; j++) {
    double sum = 0;
    for (int i = 0; i < kLatentNeurons; i++) {
      sum += latent_layer_[i] * weights_from_latent_to_output_[i][j];
    }
    output_layer_[j] = SigmoidActivation(sum);
  }

}

void AutoEncoder::BackwardPropagate() {
  for (int j = 0; j < kInputNeurons; j++) {
    gradient_[j] = (output_layer_[j] - input_layer_[j]) * (output_layer_[j]) * (1 - output_layer_[j]);
    for (int i = 0; i < kLatentNeurons; i++) {
      gradient_from_latent_to_output_[i][j] = latent_layer_[i] * gradient_[j];
    }
  }

  for (int i = 0; i < kInputNeurons; i++) {
    for (int j = 0; j < kLatentNeurons; j++) {
      gradient_of_loss_[j] += weights_from_latent_to_output_[j][i] * gradient_[i];
    }
  }

  for (int i = 0; i < kInputNeurons; i++) {
    for (int j = 0; j < kLatentNeurons; j++) {
      gradient_from_input_to_latent_[i][j] = gradient_of_loss_[j] * latent_layer_[j] * (1 - latent_layer_[j]) * input_layer_[i];
    }
  }

  // Gradient descent for encoder weights.
  for (int j = 0; j < kLatentNeurons; j++) {
    for (int i = 0; i < kInputNeurons; i++) {
      weights_from_input_to_latent_[i][j] -= learning_rate_ * gradient_from_input_to_latent_[i][j];
    }
  }

  // Gradient descent for decoder weights.
  for (int j = 0; j < kInputNeurons; j++) {
    for (int i = 0; i < kLatentNeurons; i++) {
      weights_from_latent_to_output_[i][j] -= learning_rate_ * gradient_from_latent_to_output_[i][j];

    }
  }
}

double AutoEncoder::SigmoidActivation(double x) {
  return 1.0 / (1.0 + exp(-x));
}

double AutoEncoder::GetLoss() {
  double loss = 0;
  for (int i = 0; i < kInputNeurons; i++) {
    loss += pow(input_layer_[i] - output_layer_[i], 2);
  }
  return loss;
}

void AutoEncoder::InitializeWeights() {
  // We will follow simple heuristics, initialize all weights between -1 and 1.
  std::default_random_engine generator;
  std::uniform_real_distribution<double> weights_initialization(-1, 1);

  // Initialize weights from input layer to hidden layer.
  for (int i = 0; i < kInputNeurons; i++) {
    for (int j = 0; j < kLatentNeurons; j++) {
      weights_from_input_to_latent_[i][j] = weights_initialization(generator);
    }
  }

  // Initialize weights from hidden layer to output layer.
  for (int i = 0; i < kLatentNeurons; i++) {
    for (int j = 0; j < kInputNeurons; j++) {
      weights_from_latent_to_output_[i][j] = weights_initialization(generator);
    }
  }

}

void AutoEncoder::LoadOneImage(ifstream & input_file) {
  string line;
  getline(input_file, line);
  stringstream ss(line);
  string token;

  // Since this is generative model, we don't have labels (y) any longer, just parse image's shaded information.
  int index = 0;
  while (getline(ss, token, ',')) {
    if (stoi(token) != 0) {
      input_layer_[index] = 1;
    } else {
      input_layer_[index] = 0;
    }

    index++;
  }
}

void AutoEncoder::SaveToFile() {
  ofstream model_weights(kModelWeightsFile);
  if (model_weights.is_open()) {
    for (int i = 0; i < kInputNeurons; i++) {
      for (int j = 0; j < kLatentNeurons; j++) {
        model_weights << weights_from_input_to_latent_[i][j] << " ";
      }
      model_weights << endl;
    }

    for (int i = 0; i < kLatentNeurons; i++) {
      for (int j = 0; j < kInputNeurons; j++) {
        model_weights << weights_from_latent_to_output_[i][j] << " ";
      }
      model_weights << endl;
    }

    model_weights.close();
  }

}

void AutoEncoder::LoadFromFile() {
  ifstream model_weights(kModelWeightsFile);

  if (model_weights.is_open()) {
    for (int i = 0; i < kInputNeurons; i++) {
      for (int j = 0; j < kLatentNeurons; j++) {
        model_weights >> weights_from_input_to_latent_[i][j];
      }
    }

    for (int i = 0; i < kLatentNeurons; i++) {
      for (int j = 0; j < kInputNeurons; j++) {
        model_weights >> weights_from_latent_to_output_[i][j];
      }
    }

    model_weights.close();
  }
}

void AutoEncoder::Train(ifstream & input_file) {
  if (input_file.is_open()) {

    for (int i = 0; i < kNumTrainingImages; i++) {
      LoadOneImage(input_file);
      for (int k = 0; k < epochs; k++) {
        ForwardPropagate();
        if (GetLoss() < error_threshold) {
          break;
        } else {
          BackwardPropagate();
        }
      }
    }

    input_file.close();
  }
}

} // namespace autoencoder.
