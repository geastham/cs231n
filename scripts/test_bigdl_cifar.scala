/*
 * 	Scratch Code to Test BigDL (locally) on CIFAR-10 Data
 *	======================
 *	source ~/Documents/Development/Library/BigDL/scripts/bigdl.sh 
 *	spark-shell --jars ~/Documents/Development/Library/BigDL/dl/target/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar
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
    //.setMaster("local")
    .set("spark.driver.memory", "2g")
    .set("spark.executor.memory", "2g")
    .set("spark.akka.frameSize", 64.toString)
    .set("spark.task.maxFailures", "1")
  new SparkContext(conf)
})

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
    convBNReLU(3, 64).add(Dropout((0.3)))
    convBNReLU(64, 64)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())

    convBNReLU(64, 128).add(Dropout(0.4))
    convBNReLU(128, 128)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())

    convBNReLU(128, 256).add(Dropout(0.4))
    convBNReLU(256, 256).add(Dropout(0.4))
    convBNReLU(256, 256)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())

    convBNReLU(256, 512).add(Dropout(0.4))
    convBNReLU(512, 512).add(Dropout(0.4))
    convBNReLU(512, 512)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())

    convBNReLU(512, 512).add(Dropout(0.4))
    convBNReLU(512, 512).add(Dropout(0.4))
    convBNReLU(512, 512)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())
    vggBnDo.add(View(512))

    val classifier = Sequential[Float]()
    classifier.add(Dropout(0.5))
    classifier.add(Linear(512, 512))
    classifier.add(BatchNormalization(512))
    classifier.add(ReLU(true))
    classifier.add(Dropout(0.5))
    classifier.add(Linear(512, classNum))
    classifier.add(LogSoftMax())
    vggBnDo.add(classifier)

    vggBnDo
  }
}

// Load data into RDD
val trainingImages = sc.get.textFile("file:///Users/garrett/Documents/Development/coursework/cs231n/data/cifar10-train.txt")
val testImages = sc.get.textFile("file:///Users/garrett/Documents/Development/coursework/cs231n//data/cifar10-test.txt")


// Normalizing
val trainMean = (0.4913996898739353, 0.4821584196221302, 0.44653092422369434)
val trainStd = (0.24703223517429462, 0.2434851308749409, 0.26158784442034005)
val testMean = (0.4942142913295297, 0.4851314002725445, 0.45040910258647154)
val testStd = (0.2466525177466614, 0.2428922662655766, 0.26159238066790275)

// Convert to Labeled BGR Images
val labeledTrainingImages = trainingImages.map(line => {
    // Break string into parts
    val raw_image_parts = line.split(" ")

    // Get label value
    val image_label = raw_image_parts(0).toFloat

    // Get image pixel values
    val image_data = raw_image_parts.slice(1, 3073).map(_.toFloat)

    // Create and return labeled image
    new LabeledBGRImage(image_data, 64, 64, image_label)
})

// Determine the mean and std of the distributed data
val totalImages = trainingImages.count()
val trainMeanR = labeledTrainingImages.flatMap(image => image.content.slice(0, 1024)).reduce(_ + _) / (totalImages * 1024)
val trainMeanG = labeledTrainingImages.flatMap(image => image.content.slice(1024, 2048)).reduce(_ + _) / (totalImages * 1024)
val trainMeanB = labeledTrainingImages.flatMap(image => image.content.slice(2048, 3072)).reduce(_ + _) / (totalImages * 1024)
var a = sc.get.parallelize(Array(trainMeanR))
labeledTrainingImages.map(image => (image, trainMeanR.toFloat))
val trainStdR = math.sqrt(labeledTrainingImages.map(image => image.content.slice(0, 1024).map(x => x - trainMeanR).map(x => x * x)).reduce(_ + _) / (totalImages * 1024))

val sampleLabeledTrainingImages = labeledTrainingImages._1.map { case (image: LabeledBGRImage) =>
    // Transform to sample
    Sample(
        featureTensor = Tensor(image.content, Array(64, 64, 3)),
        labelTensor = Tensor(Array(image.label), Array(1))
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
}) -> BGRImgNormalizer(trainMean, trainStd)

val sampleLabeledTestImages = labeledTestImages._1.map { case (image: LabeledBGRImage) =>
    // Transform to sample
    Sample(
        featureTensor = Tensor(image.content, Array(64, 64, 3)),
        labelTensor = Tensor(Array(image.label), Array(1))
    )
}

// Build model
val model = VggForCifar10(10)

// Setup optimizer
val optimizer = Optimizer(
    model = model,
    sampleRDD = sampleLabeledTrainingImages.randomSplit(Array(0.1, 0.9))(0),
    criterion = new ClassNLLCriterion[Float](),
    batchSize = 100
)

// Set state
val state = T("learningRate" -> 0.01, "learningRateDecay" -> 0.0002)

optimizer.setState(state)
         .setValidation(Trigger.everyEpoch, sampleLabeledTestImages.randomSplit(Array(0.2, 0.8))(0), Array(new Top1Accuracy[Float]), 100)
         .setEndWhen(Trigger.maxEpoch(20))
         .optimize()

// Stop SparkContext
sc.stop()





