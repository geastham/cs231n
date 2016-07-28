package cs231n.nn

import cs231n.data.LabeledImage

// Required Breeze libraries
import breeze.linalg._
import breeze.math._
import breeze.numerics._

/*
 *    Neural Network Classifier
 *    -------------------------
 *    Exposes two methods (train, predict) that take a training setup
 *    of images and builds a simple 2-layer neural network model and 
 *    enables that model to naively predict the class of a given image 
 *    (from predict).
 */
class NeuralNetwork {

  // Loss function for a entire dataset
  // @param x - Array of column vectors ((D + 1) x 1) of pixel values from image dataset of size N
  // @param y - Array of correct class indices (within range 0 to K)
  // @param W - trained model (W) parameters (K x (D + 1))
  // @param lambda - Double representing the regularization weight
  // @return L - calculated loss across all N data samples
  private def loss(x: Array[DenseVector[Double]], y: Array[Int], W: DenseMatrix[Double], lambda: Double): Double = {
    // To Implement
  }


  // Training function
  // @param training_images -- set of training images to train
  // @param number_of_classes -- total number of distinct classes in training / test set
  // @return W: DenseMatrix[Double] -- trained model (W) parameters - (K x (D + 1))
  def train(training_images: Array[LabeledImage], number_of_classes: Integer): DenseMatrix[Double] = {
    // To implement
  }

  // Predict -- returns predicted labels (integers)
  def predict(test_images: Array[LabeledImage]): Array[Int] = {
    // To implement
  }
}
