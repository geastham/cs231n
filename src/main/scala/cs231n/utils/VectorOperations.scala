package cs231n.utils

// Required Breeze libraries
import breeze.linalg._
import breeze.math._
import breeze.numerics._

object VectorUtils {
  // Helper -- mean -- calculates the mean of a passed in image matrix
  def mean(vectors: DenseMatrix[Array[Double]]): DenseVector[Double] = {
    // Get sum vector
    sum(vectors(*,::)) * (1.0 / vectors.length)
  }
}
