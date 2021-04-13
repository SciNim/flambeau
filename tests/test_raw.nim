import flambeau/flambeau_raw
import std/macros

{.experimental: "views".} # TODO

macro `//`*(arg: string): untyped =
  let lit = newLit("/* " & arg.strVal() & " */")
  quote do:
    {.emit: `lit`.}

type
  TensorAgreggate*[T] {.requiresinit.} = object
    raw* : RawTensor

proc TensorAgreggate*[T]() : TensorAgreggate =
  newTensorAggregate[T]()

proc newTensorAggregate[T](): TensorAgreggate[T] {.constructor, noinit.} =
  {.emit: "/* */".}

proc newTensorAggregate[T](a: RawTensor): TensorAgreggate[T] {.noinit.} =
  result = newTensorAggregate[T]()
  result.raw = a

proc `$`*[T](tensorAg: TensorAgreggate[T]) : string =
  $(tensorAg.raw)

proc initTensorAggregate*[T](raw: RawTensor) : TensorAgreggate[T] {.noinit.} =
  assign(result.raw, raw)

proc initTensorAggregate*[T](self: var TensorAgreggate[T], raw: RawTensor) =
  assign(self.raw, raw)

proc main() =
  let a = [[1, 2], [3, 4]].toRawTensor
  block:
    var b = a
    echo b
    doAssert true
  block:
    # "Create tensor"
    var rawtens : RawTensor #= initRawTensor()
    `=sink`(rawtens, a)
    echo rawtens

    var tensorAg : TensorAgreggate[int] = newTensorAggregate[int](a)
    echo sizeof(TensorAgreggate)
    echo sizeof(RawTensor)
    # var tensorAg {.noinit.} : TensorAgreggate[int] #= initTensorAggregate[int](a)
    # initTensorAggregate[int](tensorAg, a)
    # tensorAg.raw = a
    echo tensorAg.raw
    # tensorAg.raw = empty(a.sizes(), int.toScalarKind())
    # let tmp : RawTensor = from_blob(a.data_ptr(int), a.sizes(), int).clone()
    # tensorAg.raw = tmp
    doAssert true

main()
