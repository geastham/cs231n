package cs231n.cnn

import cs231n.data.LabeledImage

// BigDL imports
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.nn._

/*
 *    ===============================================================================================
 *
 *    Convolutional Neural Network Classifier
 *    -------------------------
 *    Exposes two methods (train, predict) that take a training setup
 *    of images and builds a simple 3-layer convolutional neural network model and
 *    enables that model to naively predict the class of a given image (from predict).
 *
 *    ===============================================================================================
 */

/*
 *  3-Layer Convolutional Neural Network
 *  ----------------------
 *  Private parameters that are set during training and utilized during prediction.
 */
class ConvolutionalNeuralNetwork() {

}

/*
 *  Training & Making Predictions
 *  -----------------------------
 *  A neural network knows how to accept training data (images) and perform training
 *  through stochastic gradient descent.
 */
object ConvolutionalNeuralNetwork {

  /*
   *  Training Function
   *  -----------------
   *  @param training_images -- set of training images to train
   *  @param number_of_classes -- total number of distinct classes in training / test set (for simplicity, we set the output layer to this size)
   *  @param CNN -- instantiated convolutional neural network
   *
   *  @return success -- Boolean flag determining whether traiing was successful
   */

  def train(training_images: Array[LabeledImage], number_of_classes: Int, CNN: ConvolutionalNeuralNetwork): Boolean = {


    // Return status of training
    return true
  }

  // Predict -- returns predicted labels (integers)
  def predict(test_images: Array[LabeledImage], CNN: ConvolutionalNeuralNetwork): Unit = {
    // To implement
  }

  /*
   *  Apply
   *  -----------------
   *  @param input_size -- size of input dataset (for images this will be the flattened training image)
   *  @param output_size -- total number of distinct classes in training / test set (for simplicity, we set the output layer to this size)
   */
  def apply(input_size: Integer, output_size: Integer, std: Double = 0.0001):ConvolutionalNeuralNetwork = {
    println("Initializing convolutional neural network")

    // Return new NeuralNetwork
    return new ConvolutionalNeuralNetwork()
  }
}
