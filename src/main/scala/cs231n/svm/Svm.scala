package cs231n.svm

import cs231n.data.LabeledImage

// Required Breeze libraries
import breeze.linalg._
import breeze.math._
import breeze.numerics._

/*
 *    SVM Classifier
 *    -------------------------
 *    Exposes two methods (train, predict) that take a training setup
 *    of images and builds a SVM model and enables that model
 *    to naively predict the class of a given image (from predict).
 */
class SVM {

  // Loss function for a single sample data point
  // @param x - Single column vector (D x 1) of pixel values from sample image
  // @param y - index of correct class (within range 0 to K)
  // @param W - trained model (W) parameters
  // @return Li - calculated loss across all classes for single data sample
  private def lossSingleSample(x, y, W): Double {

  }

  // Training function
  // @param training_images -- set of training images to train
  // @return W: DenseVector[Double] -- trained model (W) parameters
  def train(training_images: Array[LabeledImage]): DenseVector[Double] = {
    //
  }

  // Predict -- returns predicted labels (integers)
  def predict(test_images: Array[LabeledImage]): Array[Int] = {
    // To implement
  }
}
