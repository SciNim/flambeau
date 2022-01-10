import ../raw/bindings/[rawtensors, c10]
import ../raw/cpp/[std_cpp]
import ../raw/sugar/[interop, indexing]
import ../tensors
import std/[complex, macros]

{.experimental: "views".}
{.push inline.}

# # algorithms:
# # -----------------------------------------------------------------------
func sort*[T](self: Tensor[T], axis: int64 = -1, descending: bool = false): tuple[values: Tensor[T], originalIndices: Tensor[int64]] =
  ## Sorts the elements of the input tensor along a given dimension in ascending order by value.
  ## If dim is not given, the last dimension of the input is chosen (dim=-1).
  ## Returns (values, originalIndices) or type (TensorT, TensorInt64)
  ## where originalIndices is the original index of each values (before sorting)
  let cppSortTuple = rawtensors.sort(asRaw(self), axis, descending)
  result.values = asTensor[T](cppSortTuple.get(0))
  result.originalIndices = asTensor[int64](cppSortTuple.get(1))

func argsort*[T](self: Tensor[T], axis: int64 = -1, descending: bool = false): Tensor[int64] =
  asTensor[int64](
    rawtensors.argsort(asRaw(self), axis, descending)
  )

proc concat*[T](tensorargs: varargs[Tensor[T]], axis: int64): Tensor[T] =
  var rawVec = initCppVector[RawTensor]()
  for t in tensorargs:
    rawVec.pushBack(asRaw(t))
  let tensors = ArrayRef[RawTensor].init(rawVec)

  ## High level API for torch::cat
  result = asTensor[T](
    rawtensors.cat(tensors, axis)
  )

proc concat*[T](tensorargs: varargs[Tensor[T]]): Tensor[T] =
  # Overload because varargs + default argument don't mix well
  result = concat(tensorargs, 0.int64)

func flip*[T](self: Tensor[T], dims: openArray[int64]): Tensor[T] =
  let rawdims = dims.asTorchView()
  result = asTensor[T](
    rawtensors.flip(asRaw(self), rawdims)
  )

#
# # math
# # -----------------------------------------------------------------------
func absolute*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.absolute(asRaw(self))
  )

func abs*[T](self: Tensor[T]): Tensor[T] =
  absolute(self)

## Absolute value of Complex type is a float
func absolute*(self: Tensor[Complex32]): Tensor[float32] =
  asTensor[float32](
    rawtensors.absolute(asRaw(self))
  )

func abs*(self: Tensor[Complex32]): Tensor[float32] =
  absolute(self)

func absolute*(self: Tensor[Complex64]): Tensor[float64] =
  asTensor[float64](
    rawtensors.absolute(asRaw(self))
  )

func abs*(self: Tensor[Complex64]): Tensor[float64] =
  absolute(self)

func angle*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.angle(asRaw(self))
  )

func sgn*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.sgn(asRaw(self))
  )

func conj*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.conj(asRaw(self))
  )

func acos*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.acos(asRaw(self))
  )

func arccos*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.arccos(asRaw(self))
  )

func acosh*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.acosh(asRaw(self))
  )

func arccosh*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.arccosh(asRaw(self))
  )

func asinh*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.asinh(asRaw(self))
  )

func arcsinh*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.arcsinh(asRaw(self))
  )

func atanh*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.atanh(asRaw(self))
  )

func arctanh*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.arctanh(asRaw(self))
  )

func asin*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.asin(asRaw(self))
  )

func arcsin*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.arcsin(asRaw(self))
  )

func atan*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.atan(asRaw(self))
  )

func arctan*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.arctan(asRaw(self))
  )

func cos*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.cos(asRaw(self))
  )

func sin*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.sin(asRaw(self))
  )

func tan*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.tan(asRaw(self))
  )

func exp*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.exp(asRaw(self))
  )

func exp2*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.exp2(asRaw(self))
  )

func erf*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.erf(asRaw(self))
  )

func erfc*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.erfc(asRaw(self))
  )

func reciprocal*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.reciprocal(asRaw(self))
  )

func neg*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.neg(asRaw(self))
  )

func clamp*[T](self: Tensor[T], min, max: Scalar): Tensor[T] =
  asTensor[T](
    rawtensors.clamp(asRaw(self), min, max)
  )

func clampMin*[T](self: Tensor[T], min: Scalar): Tensor[T] =
  asTensor[T](
    rawtensors.clampMin(asRaw(self), min)
  )

func clampMax*[T](self: Tensor[T], max: Scalar): Tensor[T] =
  asTensor[T](
    rawtensors.clampMax(asRaw(self), max)
  )

func dot*[T](self: Tensor[T], other: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.dot(asRaw(self), asRaw(other))
  )

func squeeze*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](
    rawtensors.squeeze(asRaw(self))
  )

func squeeze*[T](self: Tensor[T], axis: int64): Tensor[T] =
  asTensor[T](
    rawtensors.squeeze(asRaw(self), axis)
  )

func unsqueeze*[T](self: Tensor[T], axis: int64): Tensor[T] =
  asTensor[T](
    rawtensors.unsqueeze(asRaw(self), axis)
  )

func sqrt*[T](self: Tensor[T]) : Tensor[T] =
  asTensor[T](
    rawtensors.sqrt(asRaw(self))
  )

func square*[T](self: Tensor[T]) : Tensor[T] =
  asTensor[T](
    rawtensors.square(asRaw(self))
  )

func pow*[T](self: Tensor[T], exponent: Tensor[T]) : Tensor[T] =
  asTensor[T](
    rawtensors.pow(asTensor[T](self), asTensor[T](exponent))
  )

func pow*[T](self: Tensor[T], exponent: Scalar) : Tensor[T] =
  asTensor[T](
    rawtensors.pow(asRaw(self), exponent)
  )
