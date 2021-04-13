import flambeau/flambeau_raw

{.experimental: "views".} # TODO

type
  XX* {.bycopy.}= object
    raw*: RawTensor

proc `$`*(xx: XX) : string =
  $(xx.raw)

proc main() =
  let a = [[1, 2], [3, 4]].toRawTensor
  block:
    var b = a
    echo a
    echo b
    doAssert true
  block:
    var xx : XX
    xx.raw = a
    echo xx
    doAssert true

main()
