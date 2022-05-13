#pragma once
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "sketchpad.h"
#include "../include/core/neural_network.h"
#include <vector>

using std::vector;

namespace neuralnetwork {

namespace visualizer {

/**
 * Allows a user to draw a digit on a sketchpad and uses Naive Bayes to
 * classify it.
 * The code is referenced from naive bayes mp.
 */
class NeuralNetworkApp : public ci::app::App {
 public:
  NeuralNetworkApp();

  void draw() override;
  void mouseDown(ci::app::MouseEvent event) override;
  void mouseDrag(ci::app::MouseEvent event) override;
  void keyDown(ci::app::KeyEvent event) override;

  // provided that you can see the entire UI on your screen.
  const double kWindowSize = 875;
  const double kMargin = 100;
  const size_t kImageDimension = 28;

 private:
  Sketchpad sketchpad_;
  int current_prediction_ = -1;
  NeuralNetwork model_;
};


} // namespace visualizer

}  // namespace neuralnetwork

