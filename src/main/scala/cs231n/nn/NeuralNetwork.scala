package cs231n.nn

import cs231n.data.LabeledImage

// Required Breeze libraries
import breeze.linalg._
import breeze.math._
import breeze.numerics._

/*
 *    ===============================================================================================
 *    
 *    Neural Network Classifier
 *    -------------------------
 *    Exposes two methods (train, predict) that take a training setup
 *    of images and builds a simple 2-layer neural network model and 
 *    enables that model to naively predict the class of a given image 
 *    (from predict).
 *
 *    ===============================================================================================
 */

/*
 *  2-Layer Neural Network
 *  ----------------------
 *  Private parameters that are set during training and utilized during prediction.
 */
class NeuralNetwork(
  // Layer 1
  var W_1: DenseMatrix[Double],
  var b_1: Double,

  //Layer 2
  var W_2: DenseMatrix[Double], 
  var b_2: Double) {

}

/*
 *  Training & Making Predictions
 *  -----------------------------
 *  A neural network knows how to accept training data (images) and perform training
 *  through stochastic gradient descent.
 */
object NeuralNetwork {

  /*
   *  Loss Function
   *  -------------
   *  @param x - Array of column vectors ((D + 1) x 1) of pixel values from image dataset of size N
   *  @param y - Array of correct class indices (within range 0 to K)
   *  @param W - trained model (W) parameters (K x (D + 1))
   *  @param lambda - Double representing the regularization weight
   *
   *  @return L - calculated loss across all N data samples
   */
  private def loss(x: Array[DenseVector[Double]], y: Array[Int], W: DenseMatrix[Double], lambda: Double): Double = {
    // To Implement
    return 0.0
  }

  /*
   *  Training Function
   *  -----------------
   *  @param training_images -- set of training images to train
   *  @param number_of_classes -- total number of distinct classes in training / test set (for simplicity, we set the output layer to this size)
   *
   *  @return success -- Boolean flag determining whether traiing was successful
   */
  
  def train(training_images: Array[LabeledImage], number_of_classes: Integer): Boolean = {
    // Perform Bias Trick on training data -- labels: Int, data: DenseVector[Double] - ((D + 1) x 1))
    val biased_training_data = training_images.map(i => {
      DenseVector(i.data)
    })

    // Generate training labels
    val training_labels = training_images.map(i => i.label)

    // Initialize layer matrices - W_1
    println("Initializing network layers...")
    var W_1 = DenseMatrix.rand(number_of_classes * 2, biased_training_data(0).length) * (1.0 / Math.sqrt(number_of_classes * 2.0 * biased_training_data(0).length)) 
    println(W_1)

    // Initialize layer matrices - W_2
    var W_2 = DenseMatrix.rand(number_of_classes, number_of_classes * 2) * (1.0 / Math.sqrt(number_of_classes * number_of_classes * 2.0))
    println(W_2)

    // Return status of training
    return true
  }

  // Predict -- returns predicted labels (integers)
  def predict(test_images: Array[LabeledImage]): Unit = {
    // To implement
  }

  /*
   *  Apply
   *  -----------------
   *  @param training_images -- set of training images to train
   *  @param number_of_classes -- total number of distinct classes in training / test set (for simplicity, we set the output layer to this size)
   */
   def apply(input_size: Integer, hidden_size: Integer, number_of_classes: Integer): Unit {
    
   }
}
