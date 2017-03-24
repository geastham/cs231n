/*
 * 	Scratch Code to Test BigDL (locally) on CIFAR-10 Data
 *	======================
 *	(mac)
 *  source ~/Documents/Development/Library/BigDL/scripts/bigdl.sh
 *  spark-shell --jars ~/Documents/Development/Library/BigDL/spark/dl/target/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar
 *
 *  (linux)
 *  source ~/dev/lib/BigDL/scripts/bigdl.sh
 *  sbt console -Xmx4g
 *  spark-shell --jars ~/dev/lib/BigDL/spark/dl/target/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar
 *
 */

// Apache Spark
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

// BigDL imports
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

// Reset spark context
sc.stop()

// Connect to local Spark instance
val sc = Engine.init(1, 4, true).map(conf => {
  conf.setAppName("CS 231n")
    //.setMaster("spark://green-lantern:7077")
    .setMaster("local[*]")
    .set("spark.driver.memory", "8g")
    .set("spark.executor.memory", "8g")
    .set("spark.akka.frameSize", 64.toString)
    .set("spark.task.maxFailures", "1")
  new SparkContext(conf)
})

// Set spark environment
val local_conf = new SparkConf().setAppName("CS 231n").setMaster("local[*]").set("spark.driver.memory", "8g").set("spark.executor.memory", "8g")
val conf = Engine.createSparkConf(local_conf)
val sc = new SparkContext(conf)
Engine.init

// Set singletone flag
System.setProperty("bigdl.check.singleton", false.toString)

// VGG architecture
object VggForCifar10 {
  def apply(classNum: Int): Module[Float] = {
    val vggBnDo = Sequential[Float]()

    def convBNReLU(nInputPlane: Int, nOutPutPlane: Int): Sequential[Float] = {
      vggBnDo.add(SpatialConvolution(nInputPlane, nOutPutPlane, 3, 3, 1, 1, 1, 1))
      vggBnDo.add(SpatialBatchNormalization(nOutPutPlane, 1e-3))
      vggBnDo.add(ReLU(true))
      vggBnDo
    }
    convBNReLU(3, 64).add(Dropout((0.3)))                   // (32 x 32) => (32 x 32) x 64
    convBNReLU(64, 64)                                      // (32 x 32) => (32 x 32) x 64
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())       // (32 x 32) => (16 x 16) x 64

    convBNReLU(64, 128).add(Dropout(0.4))                   // (16 x 16) => (16 x 16) x 128
    convBNReLU(128, 128)                                    // (16 x 16) => (16 x 16) x 128
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())       // (16 x 16) => (8 x 8) x 128

    convBNReLU(128, 256).add(Dropout(0.4))                  // (8 x 8) => (8 x 8) x 256
    convBNReLU(256, 256).add(Dropout(0.4))                  // (8 x 8) => (8 x 8) x 256
    convBNReLU(256, 256)                                    // (8 x 8) => (8 x 8) x 256
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())       // (8 x 8) => (4 x 4) x 256

    convBNReLU(256, 512).add(Dropout(0.4))                  // (4 x 4) => (4 x 4) x 512
    convBNReLU(512, 512).add(Dropout(0.4))                  // (4 x 4) => (4 x 4) x 512
    convBNReLU(512, 512)                                    // (4 x 4) => (4 x 4) x 512
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())       // (4 x 4) => (2 x 2) x 512

    convBNReLU(512, 512).add(Dropout(0.4))                  // (2 x 2) => (2 x 2) x 512
    convBNReLU(512, 512).add(Dropout(0.4))                  // (2 x 2) => (2 x 2) x 512
    convBNReLU(512, 512)                                    // (2 x 2) => (2 x 2) x 512
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())       // (2 x 2) => (1 x 1) x 512
    vggBnDo.add(View(512))                                  // (1 x 1) => (512 x 1)

    val classifier = Sequential[Float]()                    // (512 x 1) => (512 x 1)
    classifier.add(Dropout(0.5))                            // (512 x 1) => (512 x 1)
    classifier.add(Linear(512, 512))                        // (512 x 512) => (512 x 512)
    classifier.add(BatchNormalization(512))                 // (512 x 512) => (512 x 512)
    classifier.add(ReLU(true))                              // (512 x 512) => (512 x 512)
    classifier.add(Dropout(0.5))                            // (512 x 512) => (512 x 512)
    classifier.add(Linear(512, classNum))                   // (512 x 512) => (512 x 10)
    classifier.add(LogSoftMax())                            // (512 x 10)
    vggBnDo.add(classifier)

    vggBnDo
  }
}

object SimpleCNN {
  def apply(classNum: Int): Module[Float] = {
    val model = Sequential()
    model.add(SpatialConvolution(3, 6, 3, 3, 1, 1, 1, 1)) // (32 x 32) --> (32 x 32) x 6
      .add(Tanh())                                        // (32 x 32) --> (32 x 32) x 6
      .add(SpatialMaxPooling(2, 2, 2, 2))                 // (32 x 32) --> (16 x 16) x 6
      .add(Tanh())                                        // (16 x 16) --> (16 x 16) x 6
      .add(SpatialConvolution(6, 12, 3, 3, 1, 1, 1, 1))   // (16 x 16) --> (16 x 16) x 12
      .add(SpatialMaxPooling(2, 2, 2, 2))                 // (16 x 16) --> (8 x 8)   x 12
      .add(Reshape(Array(8 * 8 * 12)))                    // (8 x 8) --> (768 x 1)  
      .add(Linear(8 * 8 * 12, 100))                       // (768 x 1) --> (100 x 1)  
      .add(Tanh())                                        // (100 x 1) --> (100 x 1)
      .add(Linear(100, classNum))                         // (100 x 1) --> (10 x 1)
      .add(LogSoftMax())
  }
}

// Load data into RDD -- mac
val trainingImages = sc.textFile("file:///Users/garrett/Documents/Development/coursework/cs231n/data/cifar10-train.txt")
val testImages = sc.textFile("file:///Users/garrett/Documents/Development/coursework/cs231n//data/cifar10-test.txt")

// Load data into RDD - linux
val trainingImages = sc.get.textFile("file:///home/garrett/dev/coursework/cs231n/data/cifar10-train.txt")
val testImages = sc.get.textFile("file:///home/garrett/dev/coursework/cs231n//data/cifar10-test.txt")

// Normalizing
val trainMean = (125.33761, 122.97702,  113.89544)
val trainStd = (62.99322675508508, 62.08871334906125, 66.70490641235472)
val testMean = (126.05615,  123.733665, 114.88471)
val testStd = (62.89639924148415, 62.89639924148415, 66.70606331881852)

// Broadcast to executors
sc.broadcast(trainMean)
sc.broadcast(trainStd)
sc.broadcast(testMean)
sc.broadcast(testStd)

// Convert to Labeled BGR Images
val labeledTrainingImages = trainingImages.map(line => {
    // Break string into parts
    val raw_image_parts = line.split(" ")

    // Get label value
    val image_label = raw_image_parts(0).toFloat

    // Get image pixel values
    val image_data = raw_image_parts.slice(1, 3073).map(_.toFloat)

    // Create and return labeled image
    new LabeledBGRImage(image_data, 32, 32, image_label)
})

// Training - Determine the mean and std of the distributed data
val totalImages = trainingImages.count()
val trainMeanR = labeledTrainingImages.flatMap(image => image.content.slice(0, 1024)).reduce(_ + _) / (totalImages * 1024) // 125.33761
val trainMeanG = labeledTrainingImages.flatMap(image => image.content.slice(1024, 2048)).reduce(_ + _) / (totalImages * 1024) // 122.97702
val trainMeanB = labeledTrainingImages.flatMap(image => image.content.slice(2048, 3072)).reduce(_ + _) / (totalImages * 1024) // 113.89544
val trainStdR = math.sqrt(labeledTrainingImages.flatMap(image => image.content.slice(0, 1024).map(x => x - 125.33761).map(x => x * x)).reduce(_ + _) / (totalImages * 1024)) // 62.99322675508508
val trainStdG = math.sqrt(labeledTrainingImages.flatMap(image => image.content.slice(1024, 2048).map(x => x - 122.97702).map(x => x * x)).reduce(_ + _) / (totalImages * 1024)) // 62.08871334906125
val trainStdB = math.sqrt(labeledTrainingImages.flatMap(image => image.content.slice(2048, 3072).map(x => x - 113.89544).map(x => x * x)).reduce(_ + _) / (totalImages * 1024)) // 66.70490641235472

val sampleLabeledTrainingImages = labeledTrainingImages.map { case (image: LabeledBGRImage) =>
    // Center and normalize values
    val redValues = image.content.slice(0, 1024).map(r => (r - 125.33761) / 62.99322675508508).map(_.toFloat)
    val greenValues = image.content.slice(1024, 2048).map(r => (r - 122.97702) / 62.08871334906125).map(_.toFloat)
    val blueValues = image.content.slice(2048, 3072).map(r => (r - 113.89544) / 66.70490641235472).map(_.toFloat)

    // Transform to sample
    Sample(
        featureTensor = Tensor(redValues ++ greenValues ++ blueValues, Array(3, 32, 32)),
        labelTensor = Tensor(Array((image.label + 1.0).toFloat), Array(1))
    )
}

val labeledTestImages = testImages.map(line => {
    // Break string into parts
    val raw_image_parts = line.split(" ")

    // Get label value
    val image_label = raw_image_parts(0).toFloat

    // Get image pixel values
    val image_data = raw_image_parts.slice(1, 3073).map(_.toFloat)

    // Create and return labeled image
    new LabeledBGRImage(image_data, 64, 64, image_label)
})

// Test - Determine the mean and std of the distributed data
val totalImages = testImages.count()
val testMeanR = labeledTestImages.flatMap(image => image.content.slice(0, 1024)).reduce(_ + _) / (totalImages * 1024) // 126.05615
val testMeanG = labeledTestImages.flatMap(image => image.content.slice(1024, 2048)).reduce(_ + _) / (totalImages * 1024) // 123.733665
val testMeanB = labeledTestImages.flatMap(image => image.content.slice(2048, 3072)).reduce(_ + _) / (totalImages * 1024) // 114.88471
val testStdR = math.sqrt(labeledTestImages.flatMap(image => image.content.slice(0, 1024).map(x => x - 126.05615).map(x => x * x)).reduce(_ + _) / (totalImages * 1024)) //  62.89639924148415
val testStdG = math.sqrt(labeledTestImages.flatMap(image => image.content.slice(1024, 2048).map(x => x - 123.733665).map(x => x * x)).reduce(_ + _) / (totalImages * 1024)) // 61.93753229283957
val testStdB = math.sqrt(labeledTestImages.flatMap(image => image.content.slice(2048, 3072).map(x => x - 114.88471).map(x => x * x)).reduce(_ + _) / (totalImages * 1024)) // 66.70606331881852

val sampleLabeledTestImages = labeledTestImages.map { case (image: LabeledBGRImage) =>
  // Center and normalize values
  val redValues = image.content.slice(0, 1024).map(r => (r - testMean._1) / testStd._1).map(_.toFloat)
  val greenValues = image.content.slice(1024, 2048).map(r => (r - testMean._2) / testStd._2).map(_.toFloat)
  val blueValues = image.content.slice(2048, 3072).map(r => (r - testMean._3) / testStd._3).map(_.toFloat)

  // Transform to sample
  Sample(
      featureTensor = Tensor(redValues ++ greenValues ++ blueValues, Array(3, 32, 32)),
      labelTensor = Tensor(Array(image.label), Array(1))
  )
}

// Build model
val model = VggForCifar10(10)
val model = SimpleCNN(10)

// Get sample data and re-distribute
val sampleData = sampleLabeledTrainingImages.randomSplit(Array(0.5, 0.5))(0).collect()
val sampleDataRDD = sc.parallelize(sampleData, 1)

// Setup optimizer
val optimizer = Optimizer(
    model = model,
    sampleRDD = sampleLabeledTrainingImages,
    criterion = new ClassNLLCriterion[Float](),
    batchSize = 100
)

// Set state
val state = T("learningRate" -> 0.01, "learningRateDecay" -> 0.0002)

optimizer.setState(state)
         .setValidation(Trigger.everyEpoch, sampleLabeledTestImages.randomSplit(Array(0.2, 0.8))(0), Array(new Top1Accuracy[Float]), 100)
         .setEndWhen(Trigger.maxEpoch(20))
         .optimize()

optimizer.setState(state).setOptimMethod(new Adagrad()).setEndWhen(Trigger.maxEpoch(10)).optimize()

// Stop SparkContext
sc.stop()
