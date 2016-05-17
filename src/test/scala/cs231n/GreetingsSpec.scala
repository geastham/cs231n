package greeter
import org.scalatest._


class GreetingsSpec extends FlatSpec {

    "A Greeter" should "return a greeting" in {
      val greet = new Greetings
      assert(greet.greeting == "It works!")
    }

}