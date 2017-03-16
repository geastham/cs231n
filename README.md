#CS231N - Convolutional Neural Networks for Visual Recognition
This is a working project repository for assignments / explorations associated with Stanford University's [cs231n](http://cs231n.stanford.edu/) course. Please see the website for more information and project background.

I've chosen to implement this project using the Scala / Spark ecosystem with the following project dependencies:
  * Scala 2.10.5
  * ScalaTest 2.0.M5b (current ScalaTest 2.0 beta version)
  * ScalaNLP [Breeze](http://www.scalanlp.org/)
  * Apache Spark 1.6.1
  * Intel [BigDL](https://github.com/intel-analytics/BigDL)

## Quick Setup -- UI / Spark Server

1. Make sure you have a local Spark server running (both master and slave nodes) on a different WebUI port than 8080

```sh
$ ./start-master.sh --webui-port 4545
$ ./start-slave.sh spark://name.local:7077
```


2. Start the servlet repl and browse to [http://localhost:8080/](http://localhost:8080/)

```sh
$ ./sbt -mem 10000
> container:start
> browse
```

If `browse` doesn't launch your browser, manually open [http://localhost:8080/](http://localhost:8080/) in your browser.