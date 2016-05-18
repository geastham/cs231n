package cs231n.nearestneighbor

import cs231n.data.LabeledImage

// Required Breeze libraries
import breeze.linalg._
import breeze.math._
import breeze.numerics._

/*
 *    Nearest Neight Classifier
 *    -------------------------
 *    Exposes two methods (train, predict) that take a training setup
 *    of images and builds a nearest neighbor model and enables that model
 *    to naively predict the class of a given image (from predict).
 */
class NearestNeighbor {
  // Local variables
  private var trainingImages: Array[LabeledImage] = null

  // Train
  def train(training_images: Array[LabeledImage]) = {
    trainingImages = training_images // remember all input data
  }

  // Predict -- returns predicted labels (integers)
  def predict(model: Unit, test_images: Array[LabeledImage]): Array[Int] = {
    // Loop over all values in test images
    test_images.map(test_image => { // return identified label (chosen based on index from trainingImages list)
      // Calculate distances by transforming summed distance over all training images
      val distances = trainingImages.map(training_image => { // return mapped List[Integer] of computed distances between test_image and training_image
        training_image.l1Distance(test_image)
      })

      // Select the smallest distance (use argmin UFunction in Breeze)
      val minimumDistanceIndex = argmin(DenseVector(distances))

      // Return the label of the minimum image (from the trainingImage cache)
      trainingImages(minimumDistanceIndex).label
    })
  }

  def hello = "Yep, still compiles and runs!"
}
