package cs231n

import cs231n.nearestneighbor.NearestNeighbor
import cs231n.data.LabeledImage
import cs231n.utils.VectorUtils

// Required Breeze libraries
import breeze.linalg._
import breeze.math._
import breeze.numerics._


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
    val vectorizedImages = DenseMatrix(images.map(image => image.data))

    // Find the mean image vector
    val meanVectorImage = VectorUtils.mean(vectorizedImages)

    // Subtract the mean image from each labeled image and return tuple
    (meanVectorImage, images.map(i => {
      val processedRGBValues = (DenseVector(i.data) - meanVectorImage).toArray
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
  for(line <- Source.fromFile(trainingImagesFilePath).getLines().take(5000))
    trainingImages = trainingImages ++ Array(LabeledImage(line))

  // Load training data -- add to test set
  println("Loading test data...")
  for(line <- Source.fromFile(testImagesFilePath).getLines().take(1000))
    testImages = testImages ++ Array(LabeledImage(line))

  // Train classifier (NearestNeighbor)
  println("Training classifier...")
  val nn = new NearestNeighbor
  nn.train(trainingImages)

  // Predict test values
  println("Making predictions...")
  val predictions = nn.predict(testImages)

  // Show predicted accuracy
  val predictedOutcomes = testImages.map(x => x.label) zip predictions
  println("Total predicted outcomes: " + predictedOutcomes.length)
  //predictedOutcomes.foreach(println)
  val accuracy = predictedOutcomes.map(x => x match { case (actual, predicted) => if(actual == predicted) 1.0 else 0.0}).fold(0.0)(_ + _) / predictedOutcomes.length
  println(accuracy)
  println("Accuracy: " + (accuracy * 100) + "%")

}
