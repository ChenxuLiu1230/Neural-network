

#include "../../include/visualizer/neural_network_app.h"

namespace neuralnetwork {

namespace visualizer {

NeuralNetworkApp::NeuralNetworkApp()
    : sketchpad_(glm::vec2(kMargin, kMargin), kImageDimension,
                 kWindowSize - 2 * kMargin) {
  ci::app::setWindowSize((int) kWindowSize, (int) kWindowSize);
  // NeuralNetwork model;

  // Load weights from file into model.
  model_.LoadFromFile();
}


void NeuralNetworkApp::draw() {
  ci::Color8u background_color(255, 246, 148);  // light yellow
  ci::gl::clear(background_color);

  sketchpad_.Draw();

  ci::gl::drawStringCentered(
      "Press Delete to clear the sketchpad. Press Enter to make a prediction.",
      glm::vec2(kWindowSize / 2, kMargin / 2), ci::Color("black"));

  ci::gl::drawStringCentered(
      "Prediction: " + std::to_string(current_prediction_),
      glm::vec2(kWindowSize / 2, kWindowSize - kMargin / 2), ci::Color("blue"));
}

void NeuralNetworkApp::mouseDown(ci::app::MouseEvent event) {
  sketchpad_.HandleBrush(event.getPos());
}

void NeuralNetworkApp::mouseDrag(ci::app::MouseEvent event) {
  sketchpad_.HandleBrush(event.getPos());
}

void NeuralNetworkApp::keyDown(ci::app::KeyEvent event) {
  int index = 1;
  switch (event.getCode()) {
    case ci::app::KeyEvent::KEY_RETURN:
      // model_.input_out_ = sketchpad_.pixels;
      for (int i = 0; i < kImageDimension; i++) {
        for (int j = 0; j < kImageDimension; j++) {
          model_.input_out_[index] = sketchpad_.pixels[i][j] ? 1.0 : 0.0;
          index++;
        }
      }
      current_prediction_ = model_.Predict();
      break;

    case ci::app::KeyEvent::KEY_DELETE:
      sketchpad_.Clear();
      current_prediction_ = -1;
      model_.input_out_ = vector<double> (kImageDimension * kImageDimension, 0.0);
      break;
  }
}

}  // namespace visualizer

}  // namespace neuralnetwork