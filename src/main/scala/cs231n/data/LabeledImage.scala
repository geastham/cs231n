package cs231n.data

/*
 *    Labeled Image
 *    -------------------------
 *    Support class for encapsulating a labeled image. Can be instantiated by
 */

// Scala object - primarily used for built in factory method idioms
object LabeledImage {
  // factory method
  def apply(raw_image: String) = new LabeledImage
}

class LabeledImage {
  // Label accessor
  def label = "label"

  // Data accessor
  def data = "data"
}
