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
  var b_1: DenseVector[Double],

  //Layer 2
  var W_2: DenseMatrix[Double],
  var b_2: DenseVector[Double]) {

}

/*
 *  Training & Making Predictions
 *  -----------------------------
 *  A neural network knows how to accept training data (images) and perform training
 *  through stochastic gradient descent.
 */
object NeuralNetwork {

  /*
   *  Tanh Activation Function
   *  ---------------------------
   *  Private helper method around a tanh activation function to support
   *  matrix multiplication.
   */
  private def tanh(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    return 4.0 * sigmoid(m) - 1.0
  }

  /*
   *  ReLU Activation Function
   *  ------------------------
   *  Private helper method around an ReLU activation function to
   *  support matrix multiplication.
   */
  private def ReLU(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    return DenseMatrix.tabulate(m.rows, m.cols) { case (i,j) => if(m(i,j) < 0) 0.0 else m(i,j) }
  }

  /*
   *  Forward Pass Function
   *  ------------------------------------
   *  Conducts a forward pass through the neural network using the input values x
   *
   *  @param X_i -- Input vector of pixel values from image dataset
   *
   *  @return F_i -- Final output activation vector derived from the forward pass on the network at input x
   */
  private def forward_pass(X_i: DenseVector[Double]): DenseVector[Double] = {
    return DenseVector.zeros[Double](1)
  }

  /*
   *  Single Loss Function (Cross-Entropy)
   *  ------------------------------------
   *  Calculates the loss of a single input x (using Soft-Max cross entropy) model.
   *
   *  @param X_i -- Input vector of pixel values from image dataset
   *  @param F_i -- Final output activation vector derived from the forward pass on the network at input x
   *  @param y_i -- Index (Integer) of the correct class output on the K dimensional output activations F_i
   *
   *  @return Li -- Total loss (double) for a single input
   */
  private def loss_i(X_i: DenseVector[Double], F_i: DenseVector[Double], y_i: Integer): Double = {
    (-1.0 * F_i(y_i)) + log(sum(exp(F_i)))
  }

  /*
   *  Loss Function
   *  -------------
   *  @param X - Array of column vectors (D x 1) of pixel values from image dataset of size N
   *  @param Y - Array of correct class indices (within range 0 to K)
   *  @param lambda - Double representing the regularization weight
   *
   *  @return L - calculated loss across all N data samples
   */
  private def loss(X: Array[DenseVector[Double]], Y: Array[Int], lambda: Double): Double = {
    // Compute summed loss across all data points
    val data_loss = X.zipWithIndex.map(X_zipped =>
      X_zipped match {
        case (X_i, i) => loss_i(X_i, forward_pass(X_i), Y(i))
      }
    ).fold(0.0)(_ + _) * (1.0 / X.length)

    // Compute L2 regularization cost
    val regularization_loss = (0.5 * lambda * this.W_1.toDenseVector.map(w_i => w_i * w_i).fold(0.0)(_ + _)) + (0.5 * lambda *  this.W_2.toDenseVector.map(w_i => w_i * w_i).fold(0.0)(_ + _))

    // Return total loss
    data_loss + regularization_loss
  }

  /*
   *  Training Function
   *  -----------------
   *  @param training_images -- set of training images to train
   *  @param number_of_classes -- total number of distinct classes in training / test set (for simplicity, we set the output layer to this size)
   *
   *  @return success -- Boolean flag determining whether traiing was successful
   */

  def train(training_images: Array[LabeledImage]): Boolean = {
    // Perform Bias Trick on training data -- labels: Int, data: DenseVector[Double] - ((D + 1) x 1))
    val biased_training_data = training_images.map(i => {
      DenseVector(i.data)
    })

    // Generate training labels
    val training_labels = training_images.map(i => i.label)

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
   *  @param input_size -- size of input dataset (for images this will be the flattened training image)
   *  @param hidden_size -- size of intermediate (hidden) layer
   *  @param output_size -- total number of distinct classes in training / test set (for simplicity, we set the output layer to this size)
   */
  def apply(input_size: Integer, hidden_size: Integer, output_size: Integer, std: Double = 0.0001): NeuralNetwork = {
    println("Initializing network layers...")

    // 1) Initialize layer 1 matrices - W_1
    var W_1 = DenseMatrix.rand(hidden_size, input_size) * Math.sqrt(2.0 / input_size)
    println("\n1) W_1...")
    println(W_1)

    // 2) Initialize layer 1 bias -- B_1
    var b_1 = DenseVector.zeros[Double](hidden_size)
    println("\n2) b_1...")
    println(b_1)

    // 3) Initialize layer 2 matrices - W_2
    var W_2 = DenseMatrix.rand(output_size, hidden_size) * Math.sqrt(2.0 / input_size)
    println("\n3) W_2...")
    println(W_2)

    // 4) Initialize layer 2 matrixes -- B_1
    var b_2 = DenseVector.zeros[Double](output_size)
    println("\n4) b_1...")
    println(b_2)

    // Return new NeuralNetwork
    return new NeuralNetwork(W_1, b_1, W_2, b_2)
  }
}
