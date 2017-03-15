package cs231n

import cs231n.nearestneighbor.NearestNeighbor
import cs231n.svm.SVM
import cs231n.nn.NeuralNetwork
import cs231n.cnn.ConvolutionalNeuralNetwork
import cs231n.data.LabeledImage
import cs231n.utils.VectorUtils

// Required Breeze libraries
import breeze.linalg._
import breeze.math._
import breeze.numerics._
import breeze.stats._

// Data processing
import scala.io.Source

// Main method -- starts and runs the core application
object Main extends App {

  // Pre-process images -- preps an array of images for training / testing
  // @params images -- array of LabeledImage's to prep
  // return meanImage -- vector representing the mean image used to prep the data
  // return preppedImages -- array of same size with prepped images
  def preProcessImages(images: Array[LabeledImage]): (DenseVector[Double], Array[LabeledImage]) = {
    // Get the vectorized images
    val vectorizedImages = images.map(image => DenseVector(image.data)) // Array[DenseVector]

    // Find the mean image vector
    val meanVectorImage = VectorUtils.mean(vectorizedImages)

    // Subtract the mean image from each labeled image and return tuple
    (meanVectorImage, images.map(i => {
      // Center vector around mean
      val centeredVector = (DenseVector(i.data) - meanVectorImage)

      // Normalize vector
      val standardDeviation = stddev(centeredVector)
      val processedRGBValues = centeredVector.toArray.map(x => x / standardDeviation)

      new LabeledImage(i.label, processedRGBValues.slice(0, 1024), processedRGBValues.slice(1024, 2048), processedRGBValues.slice(2048, 3072))
    }))
  }

  // File paths
  val trainingImagesFilePath = new java.io.File( "." ).getCanonicalPath + "/data/cifar10-train.txt"
  val testImagesFilePath = new java.io.File( "." ).getCanonicalPath + "/data/cifar10-test.txt"

  // Data (in-memory -- not great, but can fix later with spark implementation)
  var trainingImages: Array[LabeledImage] = new Array[LabeledImage](0)
  var testImages: Array[LabeledImage] = new Array[LabeledImage](0)

  // Load training data -- add to training set
  println("Loading training data...")
  val batch_size = 100
  for(line <- Source.fromFile(trainingImagesFilePath).getLines().take(batch_size))
    trainingImages = trainingImages ++ Array(LabeledImage(line))
  val (meanTrainImage, processedTrainingImages) = preProcessImages(trainingImages)

  // Load training data -- add to test set
  println("Loading test data...")
  for(line <- Source.fromFile(testImagesFilePath).getLines().take(1))
    testImages = testImages ++ Array(LabeledImage(line))
  val (meanTestImage, processedTestImages) = preProcessImages(testImages)

  // Train classifier (NearestNeighbor)
  //println("Training classifier...")
  //val nearestneighbor = new NearestNeighbor
  //nearestneighbor.train(trainingImages, 10)

  // Train classifier (SVM)
  //println("Training classifier...")
  //val svm = new SVM
  //svm.train(trainingImages, 10)

  // Create test data (from two_layer_net.ipynb example)
  /*val sampleImage = DenseVector(16.24345364,  -6.11756414,  -5.28171752, -10.72968622)
  val sampleImages = DenseMatrix((16.24345364,  -6.11756414,  -5.28171752, -10.72968622),
                                  (8.65407629, -23.01538697,  17.44811764,  -7.61206901),
                                  (3.19039096,  -2.49370375,  14.62107937, -20.60140709))*/

  // Train classifier (Neural Network)
  println("Training Neural Network...\n")
  val nn = NeuralNetwork(processedTrainingImages(0).inputSize, processedTrainingImages(0).inputSize * 2, 10, batch_size, 0)
  //val nn = NeuralNetwork(4, 5, 3)
  NeuralNetwork.train(processedTrainingImages, 10, nn)

  // Predict test values
  /*println("Making predictions...")
  val predictions = nn.predict(testImages)

  // Show predicted accuracy
  val predictedOutcomes = testImages.map(x => x.label) zip predictions
  println("Total predicted outcomes: " + predictedOutcomes.length)
  //predictedOutcomes.foreach(println)
  val accuracy = predictedOutcomes.map(x => x match { case (actual, predicted) => if(actual == predicted) 1.0 else 0.0}).fold(0.0)(_ + _) / predictedOutcomes.length
  println(accuracy)
  println("Accuracy: " + (accuracy * 100) + "%")
  */
}
