package cs231n

import cs231n.nearestneighbor.NearestNeighbor
import cs231n.data.LabeledImage

// Required Breeze libraries
import breeze.linalg._
import breeze.math._
import breeze.numerics._
import breeze.stats.mean

// Data processing
import scala.io.Source

object Main extends App {
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
