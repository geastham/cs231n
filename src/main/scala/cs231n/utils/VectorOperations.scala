package cs231n.utils

// Required Breeze libraries
import breeze.linalg._
import breeze.math._
import breeze.numerics._

object VectorUtils {
  // Helper -- mean -- calculates the mean of a passed in array of images
  def mean(vectors: Array[DenseVector[Double]]): DenseVector[Double] = {
    // Get average
    vectors.fold(DenseVector.zeros[Double](vectors(0).length))((acc,v) => { acc + v }) * (1.0 / vectors.length)
  }
}
