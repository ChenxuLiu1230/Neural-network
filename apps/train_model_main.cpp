#include <iostream>

#include <core/neural_network.h>
#include <core/autoencoder.h>
#include <iostream>

using std::cout;
using std::endl;
using neuralnetwork::NeuralNetwork;
using autoencoder::AutoEncoder;

int main() {


    NeuralNetwork nn;
    ifstream input_file(nn.kTrainingDataFile);
    // nn.Train(input_file);

     nn.LoadFromFile();

    ifstream validation_file(nn.kValidationDataFile);
    double accuracy = nn.Validate(validation_file);
    cout << "Accuracy: " << accuracy << endl;

    return 0;
}

