import flambeau/flambeau_raw

{.experimental: "views".} # TODO

type
  TensorAgreggate* = object
    raw* : RawTensor

proc `$`*(tensorAg: TensorAgreggate) : string =
  $(tensorAg.raw)

proc main() =
  let a = [[1, 2], [3, 4]].toRawTensor
  block:
    var b = a
    echo b
    doAssert true
  block:
    var tensorAg {.noinit.} : TensorAgreggate
    tensorAg.raw = a
    echo tensorAg.raw
    # tensorAg.raw = empty(a.sizes(), int.toScalarKind())
    # let tmp : RawTensor = from_blob(a.data_ptr(int), a.sizes(), int).clone()
    # tensorAg.raw = tmp
    doAssert true

main()
