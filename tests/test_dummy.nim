import flambeau

{.experimental: "views".} # TODO

proc main() =
  # let tmp = [[-2, -6], [-12, -20]]
  # let tmp = [-2, -6, -12, -20]
  # echo tmp
  var tt : Tensor[int] = [[-2, -6], [-12, -20]].toTensor(int)
  # var tt : Tensor[int] = toTensor(tmp)
  echo tt

main()
