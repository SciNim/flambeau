import
  ../flambeau/raw_bindings/[
    data_api, tensors
  ],
  ../flambeau/cpp/std_cpp

proc main() =
  let mnist = mnist("build/mnist")

  let loader = make_data_loader(
    mnist.map(
      Stack[Example[Tensor, Tensor]].init()
    ),
    64
  )
  static: echo "loader: ", typeof(loader)

  var it = loader.start()

  echo "--- Data ---"
  it.get().data.print()
  echo "\n-----------------"
  echo "\n--- Target ---"
  it.get().target.print()
  echo "\n-----------------"

main()
