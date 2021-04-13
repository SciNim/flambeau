import flambeau

{.experimental: "views".} # TODO
proc main() =
  # let tmp = [[-2, -6], [-12, -20]]
  let tmp = [-2, -6, -12, -20]
  echo tmp
  let tt : Tensor[int] = toTensor[int](tmp)
  echo tt

main()
