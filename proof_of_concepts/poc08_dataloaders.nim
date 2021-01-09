import
  ../flambeau/raw_bindings/[
    data_api, tensors
  ]

proc main() =
  let mnist = mnist("build/mnist")

  let loader = make_data_loader(mnist, 64)
  static: echo "loader: ", typeof(loader)

main()
