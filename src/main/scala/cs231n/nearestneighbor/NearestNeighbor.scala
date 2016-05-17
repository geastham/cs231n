package cs231n.nearestneighbor

import cs231n.data.LabeledImage

/*
 *    Nearest Neight Classifier
 *    -------------------------
 *    Exposes two methods (train, predict) that take a training setup
 *    of images and builds a nearest neighbor model and enables that model
 *    to naively predict the class of a given image (from predict).
 */
class NearestNeighbor {
  // Local variables
  private var trainingImages: List[LabeledImage] = null

  // L1 Distance measure -- defers calculation to Labeled Image
  def l1Distance(x1: LabeledImage, x2: LabeledImage): Double = {
    return x1.l1Distance(x2)
  }

  // Train
  def train(training_images: List[LabeledImage]) = {
    trainImages = trainin_images // remember all input data
  }

  // Predict
  def predict(model: Unit, test_images: List[LabeledImage]) = sys.error("NearestNeighbor.predict not implmented")

  def hello = "Yep, still compiles and runs! - "
}
