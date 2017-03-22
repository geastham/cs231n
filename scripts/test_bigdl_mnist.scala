/*
 * 	Scratch Code to Test BigDL (locally) on MNIST Data
 *	======================
 *	source ~/Documents/Development/Library/BigDL/scripts/bigdl.sh 
 *	spark-shell --jars ~/Documents/Development/Library/BigDL/dl/target/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar
 */

// File loading (non-RDD)
import java.nio.file.Paths
import java.nio.ByteBuffer
import java.nio.file.{Files, Path}

// ...
import scopt.OptionParser

// Apache Spark
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

// BigDL imports
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.ByteRecord
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.image.{GreyImgNormalizer, GreyImgToBatch, BytesToGreyImg}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

// Reset spark context
sc.stop()

// Connect to local Spark instance
  val sc = Engine.init(1, 4, true).map(conf => {
    conf.setAppName("CS 231n")
      //.setMaster("local")
      .set("spark.driver.memory", "2g")
      .set("spark.executor.memory", "2g")
      .set("spark.akka.frameSize", 64.toString)
      .set("spark.task.maxFailures", "1")
    new SparkContext(conf)
  })

// Model parameters
val trainMean = 0.13066047740239506
val trainStd = 0.3081078

// LeNet5 Model
object LeNet5 {
  def apply(classNum: Int): Module[Float] = {
    val model = Sequential()
    model.add(Reshape(Array(1, 28, 28)))
      .add(SpatialConvolution(1, 6, 5, 5))
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Tanh())
      .add(SpatialConvolution(6, 12, 5, 5))
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Reshape(Array(12 * 4 * 4)))
      .add(Linear(12 * 4 * 4, 100))
      .add(Tanh())
      .add(Linear(100, classNum))
      .add(LogSoftMax())
  }
}

// Load data
def load(featureFile: Path, labelFile: Path): Array[ByteRecord] = {
    val labelBuffer = ByteBuffer.wrap(Files.readAllBytes(labelFile))
    val featureBuffer = ByteBuffer.wrap(Files.readAllBytes(featureFile))
    val labelMagicNumber = labelBuffer.getInt()

    require(labelMagicNumber == 2049)
    val featureMagicNumber = featureBuffer.getInt()
    require(featureMagicNumber == 2051)

    val labelCount = labelBuffer.getInt()
    val featureCount = featureBuffer.getInt()
    require(labelCount == featureCount)

    val rowNum = featureBuffer.getInt()
    val colNum = featureBuffer.getInt()

    val result = new Array[ByteRecord](featureCount)
    var i = 0
    while (i < featureCount) {
      val img = new Array[Byte]((rowNum * colNum))
      var y = 0
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          img(x + y * colNum) = featureBuffer.get()
          x += 1
        }
        y += 1
      }
      result(i) = ByteRecord(img, labelBuffer.get().toFloat + 1.0f)
      i += 1
    }

    result
  }


val trainData = Paths.get("/Users/garrett/Documents/Development/Data/mnist/train-images-idx3-ubyte")
val trainLabel = Paths.get("/Users/garrett/Documents/Development/Data/mnist/train-labels-idx1-ubyte")
val validationData = Paths.get("/Users/garrett/Documents/Development/Data/mnist/t10k-images-idx3-ubyte")
val validationLabel = Paths.get("/Users/garrett/Documents/Development/Data/mnist/t10k-labels-idx1-ubyte")

// Instantiate model
val model = LeNet5(classNum = 10)

// Set state
val state = T("learningRate" -> 0.001)

// Load training data
val trainSet = DataSet.array(load(trainData, trainLabel), sc.get) -> BytesToGreyImg(28, 28) -> GreyImgNormalizer(trainMean, trainStd) -> GreyImgToBatch(100)

// Setup optimizer
val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = ClassNLLCriterion[Float]())

// Load validation set
val validationSet = DataSet.array(load(validationData, validationLabel), sc.get) -> BytesToGreyImg(28, 28) -> GreyImgNormalizer(testMean, testStd) -> GreyImgToBatch(100)

// Run trainer
optimizer.setValidation(trigger = Trigger.everyEpoch, dataset = validationSet, vMethods = Array(new Top1Accuracy)).setState(state).setEndWhen(Trigger.maxEpoch(5)).optimize()


//
~/Documents/Development/Library/BigDL/scripts/bigdl.sh -- \
java -cp ~/Documents/Development/Library/BigDL/dl/target/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar \
com.intel.analytics.bigdl.models.lenet.Train \
-f ~/Documents/Development/Data/mnist/ \
--core 4 \
--node 1 \
--env local \
--checkpoint ~/Documents/Development/Data/mnist/bigdl-model \
-b 100





