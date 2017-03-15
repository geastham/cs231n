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
  var b_1: DenseMatrix[Double],

  //Layer 2
  var W_2: DenseMatrix[Double],
  var b_2: DenseMatrix[Double]) {

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
    return DenseMatrix.tabulate(m.rows, m.cols) { (i,j) => if(m(i,j) < 0) 0.0 else m(i,j) }
  }

  /*
   *  Forward Pass Function
   *  ------------------------------------
   *  Conducts a forward pass through the neural network using the input values x
   *
   *  @param X_i -- Input matrix of pixel values from image dataset
   *  @param NN - Neural network
   *
   *  @return F_i -- Final output activation vector derived from the forward pass on the network at input x
   */
  private def forward_pass(X: DenseMatrix[Double], NN: NeuralNetwork): DenseMatrix[Double] = {
    // Compute first layer activations
    val h_1 = sigmoid(X * NN.W_1 + NN.b_1)

    // Compute final activations
    var f = h_1 * NN.W_2 + NN.b_2
    //println("\n--> f")
    //println(f)

    // Return activations
    return f
  }

  /*
   *  Class Probabilities
   *  ------------------------------------
   *  Returns the normalized class probabilities computed by the provided Neural Network
   *
   *  @param X_i -- Input vector of pixel values from image dataset
   *  @param NN - Neural network
   *
   *  @return P_i -- Normalized probabilities for classes of a given input X
   */
  private def probs(X: DenseMatrix[Double], NN: NeuralNetwork): DenseMatrix[Double] = {
    // Get raw scores F
    val F = forward_pass(X, NN)

    // Raise to E
    val exp_scores = exp(F)

    // Get summed scores
    val summed_scores = sum(exp_scores, Axis._1)

    // Normalize
    val probs = DenseMatrix.tabulate(exp_scores.rows, exp_scores.cols) { (i,j) => exp_scores(i,j) / summed_scores(i) }
    //println("\n--> probs")
    //println(probs)

    // Return probabilities
    return probs
  }

  /*
   *  Gradient (Analytical)
   *  ------------------------------------
   *  Returns the derived analytical gradient updates for a given Neural Network
   *
   *  @param X_i -- Input vector of pixel values from image dataset
   *  @param y_i -- Index (Integer) of the correct class output on the K dimensional output activations F_i
   *  @param lambda - Double representing the regularization weight
   *  @param NN - Neural network
   *
   *  @return (dW_1, db_1, dW_2, db_2) -- Normalized probabilities for classes of a given input X
   */
  private def gradient(X: DenseMatrix[Double], Y: DenseVector[Int], lambda: Double, NN: NeuralNetwork): (DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double]) = {
    // [1] Determine the gradient of the SoftMax output layer (cross-entropy based) at input X
    val p = probs(X, NN)  // start with class probability scores (P_k)
    val number_of_examples = X.rows // get total number of batched examples in X -- (N x D)
    val number_of_classes = p.cols
    
    // Derive the gradient on probability scores: dL_i / dF_k = P_k - 1(y_i = k) --> (N x K)
    // * Note: We also divide by total number of examples within matrix calculation
    //         so we can encapsulate the average derivative changes within single matrix multiplications (next steps)
    //val m = DenseMatrix.tabulate(3,4) { (i,j) => i + j }
    var dscores = DenseMatrix.tabulate(number_of_examples, number_of_classes) { (i,j) => if(j == Y(i)) {
        (p(i,j) - 1.0) / number_of_examples 
      } else {
        p(i,j) / number_of_examples
      }
    }
    //println("\n--> dScores")
    //println(dscores)

    // Calculate the activations for the hidden layer (X * W_1 + b_1)
    val h_1 = sigmoid(X * NN.W_1 + NN.b_1) // --> (N x hidden_size)

    // [2] Backpropagate score gradient through second layer
    var dW_2 = h_1.t * dscores // (N x hidden_size).t * (N x K) --> (hidden_size, K)
    val summed_scores_b2 = sum(dscores, Axis._0) // (K x 1)
    val db_2 = DenseMatrix.tabulate(number_of_examples, number_of_classes) { (i,j) => summed_scores_b2(j) } // duplicate weight updates across all row examples (N x K)
    dW_2 += lambda * NN.W_2 // don't forget the regularization term (comes from global derivative on Loss function w.r.t. W_2)
    //println("\n--> dW_2")
    //println(dW_2)

    // Determine the global gradient at the hidden inputs --> (N x hidden_layers)
    //  * Notes: s(h) :* (1 - s(h)) is an elementwise multiplication
    val sigmoid_h = sigmoid(h_1)
    val sigmoid_matrix = DenseMatrix.tabulate(sigmoid_h.rows, sigmoid_h.cols) { (i,j) => sigmoid_h(i,j) * (1.0 - sigmoid_h(i,j)) } // --> (N x hidden_size)
    val dhidden_intermediate = dscores * NN.W_2.t // (N x K) * (hidden_size, K).t --> (N x hidden_size)
    val dhidden = DenseMatrix.tabulate(sigmoid_h.rows, sigmoid_h.cols) { (i,j) => dhidden_intermediate(i,j) * sigmoid_matrix(i,j) } // --> (N x hidden_layers)

    // [3] Backpropagate gradients through to first layer --> (D x hidden_size)
    var dW_1 = X.t * dhidden // (N x D).t * (N x hidden_size) --> (D x hidden_size)
    val summed_scores_b1 = sum(dhidden, Axis._0) // (hidden_size x 1)
    val db_1 = DenseMatrix.tabulate(number_of_examples, h_1.cols) { (i,j) => summed_scores_b1(j) } // duplicate weight updates across all row examples (N x K)
    dW_1 += lambda * NN.W_1 // don't forget the regularization term (comes from global derivative on Loss function w.r.t. W_1)
    //println("\n--> dW_1")
    //println(dW_1)

    // Return final values
    (dW_1, db_1, dW_2, db_2)
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
    (-1.0 * F_i(y_i.toInt)) + log(sum(exp(F_i)))
  }

  /*
   *  Loss Function
   *  -------------
   *  @param X - Array of column vectors (D x 1) of pixel values from image dataset of size N
   *  @param Y - Array of correct class indices (within range 0 to K)
   *  @param lambda - Double representing the regularization weight
   *  @param NN - Neural network
   *
   *  @return L - calculated loss across all N data samples
   */
  private def loss(X_batch: Array[DenseVector[Double]], Y: Array[Int], lambda: Double, NN: NeuralNetwork): Double = {
    // Create matrix from batch of training vectors
    val X = DenseMatrix.tabulate(X_batch.length, X_batch(0).length) { (i,j) => X_batch(i)(j) } // (N x D)

    // Get activations from forward pass
    val F = forward_pass(X, NN) // (N x K)

    // Compute summed loss across all data points
    val data_loss = X_batch.zipWithIndex.map {
      case (x_i, i) => loss_i(x_i, F.t(::,i), Y(i))
    }.fold(0.0)(_ + _) * (1.0 / X_batch.length)

    // Compute L2 regularization cost
    val regularization_loss = (0.5 * lambda * NN.W_1.toDenseVector.map(w_i => w_i * w_i).fold(0.0)(_ + _)) + (0.5 * lambda *  NN.W_2.toDenseVector.map(w_i => w_i * w_i).fold(0.0)(_ + _))

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

  def train(training_images: Array[LabeledImage], number_of_classes: Int, NN: NeuralNetwork): Boolean = {
    // Set training parameters
    val lambda = 0.001
    val learning_rate = 0.001

    // Perform Bias Trick on training data -- labels: Int, data: DenseVector[Double] - ((D + 1) x 1))
    val training_data = training_images.map(i => {
      DenseVector(i.data)
    })

    // Generate training labels
    val training_labels = training_images.map(i => i.label)
    //training_labels.map(println)

    // Calculate initial loss
    val computed_loss = loss(training_data, training_labels, lambda, NN)
    println("Computed (initial) loss: " + computed_loss)

    // Create copy of NN
    var updated_nn = new NeuralNetwork(NN.W_1, NN.b_1, NN.W_2, NN.b_2)

    // Run training (on 10 epochs)
    (0 to 50).map(i => {
      // Calculate the gradient
      val X = DenseMatrix.tabulate(training_data.length, training_data(0).length) { (i,j) => training_data(i)(j) } // (N x D)
      val Y = DenseVector(training_labels)
      val (dW_1, db_1, dW_2, db_2) = gradient(X, Y, lambda, updated_nn)

      // Perform update and create new neural network
      updated_nn = new NeuralNetwork(
        DenseMatrix.tabulate(updated_nn.W_1.rows, NN.W_1.cols) { (i,j) => updated_nn.W_1(i,j) - (learning_rate * dW_1(i,j)) },
        DenseMatrix.tabulate(updated_nn.b_1.rows, NN.b_1.cols) { (i,j) => updated_nn.b_1(i,j) - (learning_rate * db_1(i,j)) },
        DenseMatrix.tabulate(updated_nn.W_2.rows, NN.W_2.cols) { (i,j) => updated_nn.W_2(i,j) - (learning_rate * dW_2(i,j)) },
        DenseMatrix.tabulate(updated_nn.b_2.rows, NN.b_2.cols) { (i,j) => updated_nn.b_2(i,j) - (learning_rate * db_2(i,j)) }
      )

      // Calculate updated loss
      val updated_loss = loss(training_data, training_labels, lambda, updated_nn)
      println("[" + i + "] Computed (updated) loss: " + updated_loss)
    })

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
  def apply(input_size: Integer, hidden_size: Integer, output_size: Integer, batch_size: Integer, std: Double = 0.0001): NeuralNetwork = {
    println("Initializing network layers...")

    // 1) Initialize layer 1 matrices - W_1
    var W_1 = DenseMatrix.rand(input_size, hidden_size) * Math.sqrt(2.0 / input_size) // (D x hidden_size)
    //println("\n1) W_1...")
    //println(W_1)

    // 2) Initialize layer 1 bias -- B_1
    var b_1 = DenseMatrix.zeros[Double](batch_size, hidden_size) // (N x hidden_size)
    //println("\n2) b_1...")
    //println(b_1)

    // 3) Initialize layer 2 matrices - W_2
    var W_2 = DenseMatrix.rand(hidden_size, output_size) * Math.sqrt(2.0 / input_size) // (hidden_size, K)
    //println("\n3) W_2...")
    //println(W_2)

    // 4) Initialize layer 2 matrixes -- B_1
    var b_2 = DenseMatrix.zeros[Double](batch_size, output_size) // (N x K)
    //println("\n4) b_2...")
    //println(b_2)

    // Return new NeuralNetwork
    return new NeuralNetwork(W_1, b_1, W_2, b_2)
  }
}
