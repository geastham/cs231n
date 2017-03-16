name := "CS231N"

version := "0.0.1"

scalaVersion := "2.10.5"

libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.0"

libraryDependencies  ++= Seq(
  // other dependencies here
  "org.scalanlp" %% "breeze" % "0.12",
  // native libraries are not included by default. add this if you want them (as of 0.7)
  // native libraries greatly improve performance, but increase jar sizes.
  // It also packages various blas implementations, which have licenses that may or may not
  // be compatible with the Apache License. No GPL code, as best I know.
  "org.scalanlp" %% "breeze-natives" % "0.12",
  // the visualization library is distributed separately as well.
  // It depends on LGPL code.
  "org.scalanlp" %% "breeze-viz" % "0.12",

  // BigDL
  "com.intel.analytics.bigdl" % "bigdl" % "0.1.0-SNAPSHOT" from "file:///Users/Documents/Development/Library/BigDL/dl/target/bigdl-0.1.0-SNAPSHOT.jar",

  // Spark
  "org.apache.spark" % "spark-core_2.10" % "1.6.1",
  "org.apache.spark" %% "spark-mllib" % "1.6.1"
)
