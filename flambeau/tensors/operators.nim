import ../raw/bindings/[rawtensors, c10]
import ../raw/sugar/[rawinterop, indexing]
import ../tensors
import std/[complex, macros]

let t_dont_use_this {.used.} = initRawTensor()

# Operators
# -----------------------------------------------------------------------
{.push inline.}
# {.push noinit.}
func `not`*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](not asRaw(self))

func `-`*[T](self: Tensor[T]): Tensor[T] =
  asTensor[T](-asRaw(self))

func `+`*[T](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  asTensor[T](asRaw(a) + asRaw(b))

func `+`*[T](a: Tensor[T], b: SomeNumber): Tensor[T] =
  when T is SomeFloat:
    asTensor[T](asRaw(a) + b.cdouble)
  else:
    asTensor[T](asRaw(a) + b.int64)

func `+`*[T](a: SomeNumber, b: Tensor[T]): Tensor[T] =
  when T is SomeFloat:
    asTensor[T](a.cdouble + asRaw(b))
  else:
    asTensor[T](a.int64 + asRaw(b))

func `-`*[T](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  asTensor[T](asRaw(a) - asRaw(b))

func `-`*[T](a: Tensor[T], b: SomeNumber): Tensor[T] =
  # PyTorch doesn't have RawTensor - Scalar, so we use add with negative
  when T is SomeFloat:
    asTensor[T](rawtensors.add(asRaw(a), -b.cdouble))
  else:
    asTensor[T](rawtensors.add(asRaw(a), -b.int64))

func `-`*[T](a: SomeNumber, b: Tensor[T]): Tensor[T] =
  # Convert scalar to tensor and subtract
  when T is SomeFloat:
    asTensor[T](rawtensors.add(-asRaw(b), a.cdouble))
  else:
    asTensor[T](rawtensors.add(-asRaw(b), a.int64))

func `*`*[T](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  asTensor[T](asRaw(a) * asRaw(b))

func `*`*[T](a: SomeNumber, b: Tensor[T]): Tensor[T] =
  asTensor[T](a.cdouble * asRaw(b))

func `*`*[T](a: Tensor[T], b: SomeNumber): Tensor[T] =
  asTensor[T](asRaw(a) * b.cdouble)

func `/`*[T](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  asTensor[T](asRaw(a) / asRaw(b))

func `/`*[T](a: Tensor[T], b: SomeNumber): Tensor[T] =
  # Multiply by reciprocal
  when T is SomeFloat:
    asTensor[T](asRaw(a) * (1.0 / b.cdouble))
  else:
    asTensor[T](asRaw(a) * (1.0 / b.float64))

func `/`*[T](a: SomeNumber, b: Tensor[T]): Tensor[T] =
  # Create reciprocal tensor and multiply by scalar
  when T is SomeFloat:
    asTensor[T](a.cdouble * rawtensors.reciprocal(asRaw(b)))
  else:
    asTensor[T](a.float64 * rawtensors.reciprocal(asRaw(b)))

func `and`*[T](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  ## bitwise `and`.
  asTensor[T](asRaw(a) and asRaw(b))

func `or`*[T](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  ## bitwise `or`.
  asTensor[T](asRaw(a) or asRaw(b))

func `xor`*[T](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  ## bitwise `xor`.
  asTensor[T](asRaw(a) xor asRaw(b))

func bitand_mut*[T](self: var Tensor[T], s: Tensor[T]) =
  ## In-place bitwise `and`.
  rawtensors.bitand_mut(asRaw(self), asRaw(s))

func bitor_mut*[T](self: var Tensor[T], s: Tensor[T]) =
  ## In-place bitwise `or`.
  rawtensors.bitor_mut(asRaw(self), asRaw(s))

func bitxor_mut*[T](self: var Tensor[T], s: Tensor[T]) =
  ## In-place bitwise `xor`.
  rawtensors.bitxor_mut(asRaw(self), asRaw(s))

func eq*[T](a, b: Tensor[T]): Tensor[T] =
  ## Equality of each tensor values
  asTensor[T](rawtensors.eq(asRaw(a), asRaw(b)))
{.pop.}

func `+=`*[T](self: var Tensor[T], b: Tensor[T]) =
  asRaw(self) += asRaw(b)

func `+=`*[T](self: var Tensor[T], s: T) =
  asRaw(self) += s

func `-=`*[T](self: var Tensor[T], b: Tensor[T]) =
  asRaw(self) -= asRaw(b)
func `-=`*[T](self: var Tensor[T], s: T) =
  asRaw(self) -= s

func `*=`*[T](self: var Tensor[T], b: Tensor[T]) =
  asRaw(self) *= asRaw(b)

func `*=`*[T](self: var Tensor[T], s: T) =
  asRaw(self) *= s

func `/=`*[T](self: var Tensor[T], b: Tensor[T]) =
  asRaw(self) /= asRaw(b)

func `/=`*[T](self: var Tensor[T], s: T) =
  asRaw(self) /= s

func equal*[T](a, b: Tensor[T]): bool =
  equal(asRaw(a), asRaw(b))

template `==`*[T](a, b: Tensor[T]): bool =
  a.equal(b)
