import ../raw/bindings/[rawtensors, c10]
import ../raw/cpp/[std_cpp]
import ../raw/sugar/[interop, indexing]
import ../tensors
import std/[complex, macros, sugar]

{.experimental: "views".}
{.push inline, noinit.}

# # algorithms:
# # -----------------------------------------------------------------------
func sort*[T](self: Tensor[T], axis: int64 = -1, descending: bool = false): tuple[values: Tensor[T], originalIndices: Tensor[int64]] =
  ## Sorts the elements of the input tensor along a given dimension in ascending order by value.
  ## If dim is not given, the last dimension of the input is chosen (dim=-1).
  ## Returns (values, originalIndices) or type (TensorT, TensorInt64)
  ## where originalIndices is the original index of each values (before sorting)
  let cppSortTuple = rawtensors.sort(convertRawTensor(self), axis, descending)
  result.values = convertTensor[T](cppSortTuple.get(0))
  result.originalIndices = convertTensor[int64](cppSortTuple.get(1))

func argsort*[T](self: Tensor[T], axis: int64 = -1, descending: bool = false): Tensor[int64] =
  convertTensor[int64](
    rawtensors.argsort(convertRawTensor(self), axis, descending)
  )
{.pop.}

macro unpackVarargs_last(callee, arg_last: untyped; args: varargs[untyped]):untyped =
  result = newCall(callee)
  for a in args:
    result.add a
  result.add arg_last

func catImpl(tensorargs: varargs[RawTensor, convertRawTensor], axis: int64): RawTensor =
  let tensors : ArrayRef[RawTensor] = tensorargs.asTorchView()
  rawtensors.cat(tensors, axis)

template cat*[T](tensorargs: varargs[Tensor[T]], axis: int64): Tensor[T] =
  convertTensor[T](
    unpackVarargs_last(catImpl, axis, tensorargs)
  )

template cat*[T](tensorargs: varargs[Tensor[T]]): Tensor[T] =
  convertTensor[T](
    unpackVarargs_last(catImpl, 0.int64, tensorargs)
  )

{.push inline, noinit.}
func flip*[T](self: Tensor[T], dims: openArray[int64]): Tensor[T] =
  let rawdims = dims.asTorchView()
  convertTensor[T](
    rawtensors.flip(convertRawTensor(self), rawdims)
  )

#
# # math
# # -----------------------------------------------------------------------
func abs*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.abs(convertRawTensor(self))
  )

func absolute*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.absolute(convertRawTensor(self))
  )

## Absolute value of Complex type is a float
func abs*(self: Tensor[Complex32]): Tensor[float32] =
  convertTensor[float32](
    rawtensors.abs(convertRawTensor(self))
  )

func absolute*(self: Tensor[Complex32]): Tensor[float32] =
  convertTensor[float32](
    rawtensors.absolute(convertRawTensor(self))
  )

func abs*(self: Tensor[Complex64]): Tensor[float64] =
  convertTensor[float64](
    rawtensors.abs(convertRawTensor(self))
  )

func absolute*(self: Tensor[Complex64]): Tensor[float64] =
  convertTensor[float64](
    rawtensors.absolute(convertRawTensor(self))
  )

func angle*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.angle(convertRawTensor(self))
  )

func sgn*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.sgn(convertRawTensor(self))
  )

func conj*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.conj(convertRawTensor(self))
  )

func acos*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.acos(convertRawTensor(self))
  )

func arccos*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.arccos(convertRawTensor(self))
  )

func acosh*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.acosh(convertRawTensor(self))
  )

func arccosh*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.arccosh(convertRawTensor(self))
  )

func asinh*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.asinh(convertRawTensor(self))
  )

func arcsinh*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.arcsinh(convertRawTensor(self))
  )

func atanh*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.atanh(convertRawTensor(self))
  )

func arctanh*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.arctanh(convertRawTensor(self))
  )

func asin*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.asin(convertRawTensor(self))
  )

func arcsin*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.arcsin(convertRawTensor(self))
  )

func atan*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.atan(convertRawTensor(self))
  )

func arctan*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.arctan(convertRawTensor(self))
  )

func cos*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.cos(convertRawTensor(self))
  )

func sin*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.sin(convertRawTensor(self))
  )

func tan*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.tan(convertRawTensor(self))
  )

func exp*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.exp(convertRawTensor(self))
  )

func exp2*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.exp2(convertRawTensor(self))
  )

func erf*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.erf(convertRawTensor(self))
  )

func erfc*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.erfc(convertRawTensor(self))
  )

func reciprocal*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.reciprocal(convertRawTensor(self))
  )

func neg*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.neg(convertRawTensor(self))
  )

func clamp*[T](self: Tensor[T], min, max: Scalar): Tensor[T] =
  convertTensor[T](
    rawtensors.clamp(convertRawTensor(self), min, max)
  )

func clampMin*[T](self: Tensor[T], min: Scalar): Tensor[T] =
  convertTensor[T](
    rawtensors.clampMin(convertRawTensor(self), min)
  )

func clampMax*[T](self: Tensor[T], max: Scalar): Tensor[T] =
  convertTensor[T](
    rawtensors.clampMax(convertRawTensor(self), max)
  )

func dot*[T](self: Tensor[T], other: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.dot(convertRawTensor(self), convertRawTensor(other))
  )

func squeeze*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](
    rawtensors.squeeze(convertRawTensor(self))
  )

func squeeze*[T](self: Tensor[T], axis: int64): Tensor[T] =
  convertTensor[T](
    rawtensors.squeeze(convertRawTensor(self), axis)
  )

func unsqueeze*[T](self: Tensor[T], axis: int64): Tensor[T] =
  convertTensor[T](
    rawtensors.unsqueeze(convertRawTensor(self), axis)
  )

func sqrt*[T](self: Tensor[T]) : Tensor[T] =
  convertTensor[T](
    rawtensors.sqrt(convertRawTensor(self))
  )

func square*[T](self: Tensor[T]) : Tensor[T] =
  convertTensor[T](
    rawtensors.square(convertRawTensor(self))
  )

func pow*[T](self: Tensor[T], exponent: Tensor[T]) : Tensor[T] =
  convertTensor[T](
    rawtensors.pow(convertTensor[T](self), convertTensor[T](exponent))
  )

func pow*[T](self: Tensor[T], exponent: Scalar) : Tensor[T] =
  convertTensor[T](
    rawtensors.pow(convertRawTensor(self), exponent)
  )
