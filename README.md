# Neural Network implemented in C++
This project implements some classical machine learning models in C++. It has 2 parts,
the first part is a nueral network with 1 hidden layer, the second part is an auto-encoder
used to compress data into lower dimensions for the benefit of smaller cost in training time 
and storage memory. 

The neural network performs well on the MNIST dataset (a famous dataset containing handwritten
digits), achieving about 95% accuracy on the validation set. The dataset in txt format 
is sourced from halimb, which is mentioned in the reference. Users can run "train-model" to
validate the accuracy of the model. In addition, the neural network is connected to a Cinder
sketchpad, users can run "sketchpad-classifier" and draw a handwritten digit on it and click "keyDown" or Enter to call the model
to do the prediction, and the user can see the prediction results at the bottom of the sketchpad.

For auto-encoder, feel free to change the size of the latent layer (the size of compressed data, hyperparameter), 
which is kLatentNeurons in autoencoder.h. For now it is set at 196 (14 * 14, half of the original 28 * 28 image).

Reference:
Credited to halimb, the author of MNIST in txt format. 
Data source: https://github.com/halimb/MNIST-txt