package cs231n.data

import breeze.linalg._

/*
 *    Labeled Image
 *    -------------------------
 *    Support class for encapsulating a labeled image. Can only be instantiated from ImageParser API.
 */

// Scala object - primarily used for built in factory method idioms
object LabeledImage {
  // factory method
  def apply(raw_image: String) = new LabeledImage(0, List(0), List(0), List(0))
}

class LabeledImage(label: Integer, redValues: List[Integer], greenValues: List[Integer], blueValues: List[Integer]) {

  // L1 Distance function
  def l1Distance(image: LabeledImage): Double = {

  }

  // Label accessor
  def label: Integer = label // return instantiated label

  // Data accessor
  def data = "data"
}
