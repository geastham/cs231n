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
  // @param x - Single column vector ((D + 1) x 1) of pixel values from sample image
  // @param y - index of correct class (within range 0 to K)
  // @param W - trained model (W) parameters (K x (D + 1))
  // @return Li - calculated loss across all classes for single data sample
  private def loss_single_sample(x: DenseVector[Double], y: Integer, W: DenseMatrix[Double]): Double = {
    // Set delta
    val delta: Double = 1.0

    // Calculate dot product of W (K x (D + 1)) and x ((D + 1) x 1)
    val scores: DenseVector[Double] = W * x

    // Keep track of score(y) - (K x 1)
    val score_y = scores(y.toInt)

    // Compute the margins for all K classes
    val margins = scores.toArray.map(k_score => {
      if(k_score - score_y + delta > 0) {
        k_score - k_score + delta
      } else {
        0.0
      }
    })

    // Remove the computed margin for case y = j
    margins(y) = 0.0

    // Sum over all computed margins and return
    sum(margins)
  }

  // Loss function for a entire dataset
  // @param x - Array of column vectors ((D + 1) x 1) of pixel values from image dataset of size N
  // @param y - Array of correct class indices (within range 0 to K)
  // @param W - trained model (W) parameters (K x (D + 1))
  // @return L - calculated loss across all N data samples
  private def loss(x: Array[DenseVector[Double]], y: Array[Int], W: DenseMatrix[Double], lambda: Double): Double = {
    // Compute summed loss across all data points
    val data_loss = x.zipWithIndex.map(X_zipped =>
      X_zipped match {
        case (x_i, i) => loss_single_sample(x_i, y(i), W)
      }
    ).fold(0.0)(_ + _) * (1.0 / x.length)

    // Compute L2 regularization cost
    val regularization_loss = lambda * W.toDenseVector.map(w_i => w_i * w_i).fold(0.0)(_ + _)

    // return raw data loss
    data_loss + regularization_loss
  }

  // Numerical gradient (for gradient checking)
  // @param f - curried closure over loss function
  // @param W - trained model (W) parameters (K x (D + 1))
  // @return dW - the numerically derived gradient vector at value x (K X (D + 1))
  private def numerical_gradient(f: (DenseMatrix[Double]) => Double, W: DenseMatrix[Double]): DenseMatrix[Double] = {
    // Initialize local variables
    var x = W.copy // make an editable copy of W
    var fx = f(x) // Get loss value at original W
    var gradient = DenseMatrix.zeros[Double](x.rows, x.cols) // Initialize random gradient
    var h = 0.0000001 // arbitrary small h (to simulate limit)

    // Iterate over all values of W -- change one at a time
    (0 to (x.rows - 1)).foreach(i => {
      (0 to (x.cols - 1)).foreach(j => {
        // Evaluate function at W + h
        val original_W_i_j = x(i,j)
        x(i,j) = original_W_i_j + h // increment by h
        val fxh = f(x) // evaluate f(x + h)
        x(i,j) = original_W_i_j // restore previous value

        // Compute the partial derivative
        gradient(i,j) = (fxh - fx) / h
      })
    })

    // Return the calculated gradient
    gradient
  }

  // Training function
  // @param training_images -- set of training images to train
  // @param number_of_classes -- total number of distinct classes in training / test set
  // @return W: DenseMatrix[Double] -- trained model (W) parameters - (K x (D + 1))
  def train(training_images: Array[LabeledImage], number_of_classes: Integer): DenseMatrix[Double] = {
    // Perform Bias Trick on training data -- labels: Int, data: DenseVector[Double] - ((D + 1) x 1))
    val biased_training_data = training_images.map(i => {
      DenseVector(i.data ++ Array(1.0))
    })

    // Generate training labels
    val training_labels = training_images.map(i => i.label)

    // Initialize random W
    val W: DenseMatrix[Double] = DenseMatrix.rand(number_of_classes, biased_training_data(0).length)
    println(W)

    // Calculate loss on single run of data
    val total_loss = loss(biased_training_data, training_labels, W, 0.001)
    println("Total loss from first run: " + total_loss)

    // Create closure over loss function
    def loss_closure(_w: DenseMatrix[Double]): Double = {
      loss(biased_training_data, training_labels, _w, 0.001)
    }

    // Calculate and print numerical gradient from initial W
    val gradient = numerical_gradient(loss_closure, W)
    println(gradient)

    // Try loss function with different step sizes
    (-10 to 3).foreach(i => {
      // Set step size
      val step_size = Math.pow(10, i)

      // Update new W with gradient
      val W_i: DenseMatrix[Double] = W - step_size * gradient

      // Calculate new loss function
      val loss_new = loss(biased_training_data, training_labels, W_i, 0.001)

      // Print output
      println("New loss after gradient update [" + i + "]:" + loss_new)
    })

    // Return trained weight matrix W
    W
  }

  // Predict -- returns predicted labels (integers)
  //def predict(test_images: Array[LabeledImage]): Array[Int] = {
    // To implement
  //}
}
