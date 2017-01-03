package cs231n.cnn

import cs231n.data.LabeledImage

// Import MXNet dependencies
import ml.dmlc.mxnet._
import ml.dmlc.mxnet.optimizer.SGD

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

  // LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
  // Haffner. "Gradient-based learning applied to document recognition."
  // Proceedings of the IEEE (1998)
  def getLenetNetwork: Symbol = {
    val data = Symbol.Variable("data")
    // first conv
    val conv1 = Symbol.Convolution()(Map("data" -> data, "kernel" -> "(5, 5)", "num_filter" -> 20))
    val tanh1 = Symbol.Activation()(Map("data" -> conv1, "act_type" -> "tanh"))
    val pool1 = Symbol.Pooling()(Map("data" -> tanh1, "pool_type" -> "max",
                                     "kernel" -> "(2, 2)", "stride" -> "(2, 2)"))
    // second conv
    val conv2 = Symbol.Convolution()(Map("data" -> pool1, "kernel" -> "(5, 5)", "num_filter" -> 50))
    val tanh2 = Symbol.Activation()(Map("data" -> conv2, "act_type" -> "tanh"))
    val pool2 = Symbol.Pooling()(Map("data" -> tanh2, "pool_type" -> "max",
                                     "kernel" -> "(2, 2)", "stride" -> "(2, 2)"))
    // first fullc
    val flatten = Symbol.Flatten()(Map("data" -> pool2))
    val fc1 = Symbol.FullyConnected()(Map("data" -> flatten, "num_hidden" -> 500))
    val tanh3 = Symbol.Activation()(Map("data" -> fc1, "act_type" -> "tanh"))
    // second fullc
    val fc2 = Symbol.FullyConnected()(Map("data" -> tanh3, "num_hidden" -> 10))
    // loss
    val lenet = Symbol.SoftmaxOutput(name = "softmax")(Map("data" -> fc2))
    lenet
  }

  // Wraper to setup model
  def setupModel(numExamples: Int, batchSize: Int, numEpochs: Int): FeedForward = {
    // set epoch size
    val epochSize = numExamples / batchSize

    // set learnig rate factor
    val lrScheduler = new FactorScheduler(step = Math.max((epochSize * 1f).toInt, 1), factor = 1f)

    // Use SGD optimizer
    val optimizer = new SGD(learningRate = 0.1f, lrScheduler = lrScheduler, clipGradient = 0f, momentum = 0.9f, wd = 0.00001f)

    // Setup model for training
    return new FeedForward(ctx = Array(Context.gpu(0)),
                            symbol = getLenetNetwork,
                            numEpoch = numEpochs,
                            optimizer = optimizer,
                            initializer = new Xavier(factorType = "in", magnitude = 2.34f),
                            //argParams = argParams,
                            //auxParams = auxParams,
                            //beginEpoch = beginEpoch,
                            epochSize = epochSize)
  }
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
