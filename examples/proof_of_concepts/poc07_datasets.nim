import ../flambeau_nn

proc main() =
  let mnist = mnist("build/mnist")

  echo "Data"
  mnist.get(0).data.print()
  echo "\n-----------------------"
  echo "Target"
  mnist.get(0).target.print()
  echo "\n-----------------------"

  echo "Stateful dataset? ", mnist.typeof.is_stateful

main()
