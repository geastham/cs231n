#Scala Starter Project using SBT and ScalaTest
For me, one of the hardest parts about learning a new language is figuring out how to get
the correct environment setup where I can run tests. This is just a simple Scala project
using SBT and [ScalaTest](http://www.scalatest.org/).

I found this blog post useful for getting setup: [Quick Sbt Tutorial](http://grosdim.blogspot.com/2013/01/quick-sbt-tutorial.html)

This project is configured to use:
  * Scala 2.10.0
  * ScalaTest 2.0.M5b (current ScalaTest 2.0 beta version)

## Quick Setup

 1. Install [SBT](http://www.scala-sbt.org/)
 1. Clone this project
 1. Run `sbt` (This will also download all dependencies)
 1. In the sbt console you can:
  * `run` - runs the greeter.Hello application
  * `test` - runs the GreetingsSpec tests
  * `console` - run the REPL with your project environment loaded
  * `compile` - compiles. :)
  * `clean` - cleans the generated file from compiling

### IntelliJ Users
I use IntelliJ and I use [sbt-idea](https://github.com/mpeltonen/sbt-idea) to generate the
Intellij project files. When you have it setup as described in their readme you can run
`gen-idea` in your sbt console to generate the .idea project folders.

### Example Console Use
When you run the `console` in sbt your projects environment is available. For example, you can
create an instance of Greetings and access its methods:

```
> console
[info] Starting scala interpreter...
[info]
Welcome to Scala version 2.10.0 (Java HotSpot(TM) 64-Bit Server VM, Java 1.6.0_37).
Type in expressions to have them evaluated.
Type :help for more information.

scala> val g = new greeter.Greetings
g: greeter.Greetings = greeter.Greetings@304d03e5

scala> g.greeting
res0: String = It works!
```