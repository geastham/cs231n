package cs231n.nearestneighbor

/*
 *    Nearest Neight Classifier
 *    -------------------------
 *    Exposes two methods (train, predict) that take a training setup
 *    of images and builds a nearest neighbor model and enables that model
 *    to naively predict the class of a given image (from predict).
 */
class NearestNeighbor {
  // Train
  def train(train_images: Unit, train_labels: Unit) = sys.error("NearestNeighbor.train not implmented")

  // Predict
  def predict(model: Unit, test_images: Unit) = sys.error("NearestNeighbor.predict not implmented")

  def hello = "Yep, still compiles and runs!"
}
