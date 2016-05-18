package cs231n.data

/*
 *    Labeled Image
 *    -------------------------
 *    Support class for encapsulating a labeled image. Can only be instantiated from ImageParser API.
 */

// Scala object - primarily used for built in factory method idioms
object LabeledImage {
  // factory method
  def apply(raw_image: String) = {
    // break string into parts
    val raw_image_parts = raw_image.split(" ")

    // 
  }
}

class LabeledImage(label: Integer, redValues: List[Integer], greenValues: List[Integer], blueValues: List[Integer]) {

  // L1 Distance function --
  def l1Distance(image: LabeledImage): Double = {
    // Grab image data (to cache in local clojure)
    val imageData = image.data

    // Zip image distances together with absolute difference
    this.data.zipWithIndex.map((pixelValue, index) => {
      Math.abs(image.data(index) - pixelValue)
    }).fold(0.0)((acc,v) => { acc + v }) // return folded sum of absolute values
  }

  // Label accessor
  def label: Integer = label // return instantiated label

  // Data vector -- used in
  def data: List[Integer] = redValues :: greenValues :: blueValue
}
