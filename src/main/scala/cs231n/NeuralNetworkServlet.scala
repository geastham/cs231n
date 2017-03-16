package cs231n.NeuralNetworkServlet

// Use scalatra to define routes
import org.scalatra._

// Use Spark
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

// Scala utils
import scala.util.Try

// Create SearchServlet
class NeuralNetworkServlet extends ScalatraServlet {

  // Initialize Spark context.
  // http://spark.apache.org/docs/1.2.0/programming-guide.html#initializing-spark
  val conf = new SparkConf().setAppName("CS231n").setMaster("local").set("spark.driver.memory", "2g").set("spark.executor.memory", "2g")
  val sc = new SparkContext(conf)

  // Set SQL Context
  val sqlContext = new org.apache.spark.sql.SQLContext(sc)

  // Spark SQL imports
  import sqlContext.implicits._
  import org.apache.spark.sql.SchemaRDD

  // This defines the HTTP GET method that invokes Spark.
  get("/hello/:name") {

    // Get name from request
    var name = params("name")

    // Manually construct JSON response
    var jsonResponse = """{"hello": """
    jsonResponse += """"""" + name + """"}"""

    // Set return type
    contentType="application/json"
    
    // Return result
    {jsonResponse}

  }

  // Serve static files.
  notFound {
    contentType = "text/html"
    serveStaticResource() getOrElse resourceNotFound()
  }

}
