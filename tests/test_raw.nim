import flambeau/flambeau_raw

{.experimental: "views".} # TODO

type
  TensorAgreggate* {.bycopy.}= object
    raw*: RawTensor

proc `$`*(tensorAg: TensorAgreggate) : string =
  $(tensorAg.raw)

proc main() =
  let a = [[1, 2], [3, 4]].toRawTensor
  block:
    var b = a
    doAssert true
  block:
    var tensorAg : TensorAgreggate
    # tensorAg.raw = a
    tensorAg.raw = empty(a.sizes(), int.toScalarKind())
    let tmp : RawTensor = from_blob(a.data_ptr(int), a.sizes(), int).clone()
    tensorAg.raw = tmp
    doAssert true

main()
