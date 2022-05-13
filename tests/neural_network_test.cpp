#include <catch2/catch.hpp>
#include <core/neural_network.h>
#include <string>

using neuralnetwork::NeuralNetwork;

static const double epsilon = 0.005;
static const string training_set = "/Users/479846747qq.com/Desktop/Cinder/cinder_0.9.2_mac/my-projects/final-project-ChenxuLiu1230/data/MNIST_train.txt";

TEST_CASE("Test for loading images") {
  SECTION("Test loading data from file into model") {
    NeuralNetwork nn;
    ifstream input_file(training_set);
    nn.LoadOneImage(input_file);
    REQUIRE(nn.label_[6] == Approx(1.0).margin(epsilon));
    REQUIRE(nn.input_out_[1] == Approx(0.0).margin(epsilon));
    REQUIRE(nn.input_out_[153] == Approx(1.0).margin(epsilon));
  }
}

TEST_CASE("Test for utility functions") {
  NeuralNetwork nn;
  ifstream input_file(training_set);
  SECTION("Test sigmoid activation works normally.") {
    REQUIRE(nn.SigmoidActivation(1) == Approx(0.731).margin(epsilon));
  }

  SECTION("Test squared loss is calculated correctly") {
    nn.LoadOneImage(input_file);
    nn.ForwardPropagation();
    REQUIRE(nn.GetLoss() == Approx(2.946).margin(0.005));
  }
}

TEST_CASE("Test for forward propagation") {
  NeuralNetwork nn;
  ifstream input_file(training_set);
  nn.LoadOneImage(input_file);
  nn.ForwardPropagation();

  SECTION("Test forwarding from input to hidden layer works well.") {
    REQUIRE(nn.hidden_in_[1] == Approx(9.536).margin(epsilon));
    REQUIRE(nn.hidden_in_[128] == Approx(-2.967).margin(epsilon));
  }

  SECTION("Test forwarding from hidden to output layer works well.") {
    REQUIRE(nn.output_in_[1] == Approx(5.349).margin(epsilon));
    REQUIRE(nn.output_in_[10] == Approx(-1.947).margin(epsilon));
  }
}

TEST_CASE("Test for backward propagation") {
  NeuralNetwork nn;
  ifstream input_file(training_set);
  nn.LoadOneImage(input_file);
  nn.ForwardPropagation();
  nn.BackwardPropagation();
  SECTION("Test that gradient of loss with respect to weights from input to hidden layer is calculated correctly.") {
    REQUIRE(nn.weight_from_input_to_hidden_[1][1] == Approx(-0.83).margin(epsilon));
    REQUIRE(nn.weight_from_input_to_hidden_[784][128] == Approx(-0.159).margin(epsilon));
  }

  SECTION("Test that gradient of loss with respect to weights from hidden to output layer is calculated correctly.") {
    REQUIRE(nn.weight_from_hidden_to_output_[1][1] == Approx(0.385).margin(epsilon));
    REQUIRE(nn.weight_from_hidden_to_output_[128][10] == Approx(-0.235).margin(epsilon));
  }
}

TEST_CASE("Test for loading weights from file into model") {
  SECTION("Test loading weights") {
    NeuralNetwork nn;
    nn.LoadFromFile();
    REQUIRE(nn.weight_from_input_to_hidden_[1][1] == Approx(-0.4).margin(epsilon));
    REQUIRE(nn.weight_from_hidden_to_output_[128][10] == Approx(-0.590).margin(epsilon));
  }
}