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
  // @param W - trained model (W) parameters (K x D)
  // @return Li - calculated loss across all classes for single data sample
  private def lossSingleSample(x: DenseVector[Double], y: Integer, W: DenseMatrix[Double]): Double = {
    // Set delta
    val delta: Double = 1.0

    // Calculate dot product of W (K x D) and x (D x 1)
    val scores: DenseVector[Double] = W * x

    // Compute the margins for all classes
    val margins = scores.map(s => if(s - s(y) + delta > 0) s - s(y) + delta else 0.0)

    // Remove the computed margin for case y = j
    margins(y) = 0.0

    // Sum over all computed margins and return
    sum(margins)
  }

  // Training function
  // @param training_images -- set of training images to train
  // @return W: DenseMatrix[Double] -- trained model (W) parameters
  def train(training_images: Array[LabeledImage]): DenseMatrix[Double] = {
    //
  }

  // Predict -- returns predicted labels (integers)
  def predict(test_images: Array[LabeledImage]): Array[Int] = {
    // To implement
  }
}
