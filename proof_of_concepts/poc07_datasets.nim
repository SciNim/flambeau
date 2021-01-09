import
  ../flambeau/raw_bindings/[
    data_api, tensors
  ]

proc main() =
  let mnist = mnist("build/mnist")

  echo "Data"
  mnist.get(0).data.print()
  echo "\n-----------------------"
  echo "Target"
  mnist.get(0).target.print()
  echo "\n-----------------------"

main()
