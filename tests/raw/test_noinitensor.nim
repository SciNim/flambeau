import flambeau/flambeau_raw
import std/[macros, unittest]

{.experimental: "views".} # TODO

macro `//`*(arg: string): untyped =
  let lit = newLit("/* " & arg.strVal() & " */")
  quote do:
    {.emit: `lit`.}

type
  TensorAgreggate*[T] = distinct RawTensor

  CompositeTensor* = object
    t : TensorAgreggate[int]

proc newTensorAggregate[T](): TensorAgreggate[T] =
  RawTensor(result) = initRawTensor()

proc newTensorAggregate[T](a: RawTensor): TensorAgreggate[T] =
  RawTensor(result) = a

proc `$`*[T](tensorAg: TensorAgreggate[T]): string =
  $(RawTensor(tensorAg))

template toTensorAgImpl[T: SomeTorchType](a: untyped) : TensorAgreggate[T] =
  TensorAgreggate[T](toRawTensor(a))

func toTensorAg*[T: SomeTorchType](oa: openArray[T]): TensorAgreggate[T]  =
  return toTensorAgImpl[T](oa)

func toTensorAg*[T: seq|array](oa: openArray[T]): auto  =
  type V = getBaseType(T)
  return toTensorAgImpl[V](oa)

func `==`*[T](lhs, rhs: TensorAgreggate[T]) : bool =
  RawTensor(lhs) == RawTensor(rhs)

proc main() =
  suite "RawTensor Initialization and constraint":
    let a = [[1, 2], [3, 4]].toRawTensor()
    test "Assignment":
      var b = a
      check b == a
      check $(b) == $(a)

    test "Raw Memory handling":
      # "Create tensor"
      var rawtens: RawTensor = initRawTensor()
      let memdata = cast[ptr UncheckedArray[uint64]](rawtens.unsafeAddr)
      # Show casing that modifying the memdata[0] triggers the refcount
      let m = memdata[0]
      zeroMem(rawtens.unsafeAddr, sizeof(rawtens))
      # If this line is commentend, the line rawtens = a will detect a reference counting equal to zero and destroy the pointers before the assignment operator (causing a segfault)
      # This behaviour of zero-ing memory causing a ref. count is the reason why the {.noinit.} is needed on proc that return a Tensor
      memdata[0] = m # Comment to create a segfault
      rawtens = a
      check: rawtens == a
      check: $(rawtens) == $(a)

    test "Tensor Aggregate":
      var tensorAg: TensorAgreggate[int] = newTensorAggregate[int](a)
      echo tensorAg

      check: RawTensor(tensorAg) == a
      check: $(tensorAg) == $(a)
      let b = [[1, 2], [3, 4]].toTensorAg()
      echo b
      check: tensorAg == b

    test "Tensor Aggregate assignment":
      var tensorAg : TensorAgreggate[int]
      RawTensor(tensorAg) = a

      check: RawTensor(tensorAg) == a
      check: $(tensorAg) == $(a)

      var comp : CompositeTensor #= new(CompositeTensor)
      # If CompsiteTensor is a ref, the next line segfault
      comp.t = tensorAg
      echo comp.t
      check: RawTensor(comp.t) == a

    test "toTensorAg":
      let b = [[1, 2], [3, 4]].toTensorAg()
      check: RawTensor(b) == a
      echo b

main()
