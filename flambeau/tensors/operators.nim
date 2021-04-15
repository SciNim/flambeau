import ../raw/bindings/[rawtensors, c10]
import ../raw/cpp/[std_cpp]
import ../raw/sugar/[interop, indexing]
import ../tensors
import std/[complex, macros]

# Operators
# -----------------------------------------------------------------------
{.push inline.}
{.push noinit.}
func `not`*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](not convertRawTensor(self))

func `-`*[T](self: Tensor[T]): Tensor[T] =
  convertTensor[T](-convertRawTensor(self))

func `+`*[T](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  convertTensor[T](convertRawTensor(a) + convertRawTensor(b))

func `-`*[T](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  convertTensor[T](convertRawTensor(a) - convertRawTensor(b))

func `*`*[T](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  convertTensor[T](convertRawTensor(a) * convertRawTensor(b))

func `*`*[T](a: SomeNumber, b: Tensor[T]): Tensor[T] =
  convertTensor[T](a.cdouble * convertRawTensor(b))

func `*`*[T](a: Tensor[T], b: SomeNumber): Tensor[T] =
  convertTensor[T](convertRawTensor(a) * b.cdouble)

func `and`*[T](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  ## bitwise `and`.
  convertTensor[T](convertRawTensor(a) and convertRawTensor(b))

func `or`*[T](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  ## bitwise `or`.
  convertTensor[T](convertRawTensor(a) or convertRawTensor(b))

func `xor`*[T](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  ## bitwise `xor`.
  convertTensor[T](convertRawTensor(a) xor convertRawTensor(b))

func bitand_mut*[T](self: var Tensor[T], s: Tensor[T]) =
  ## In-place bitwise `and`.
  rawtensors.bitand_mut(convertRawTensor(self), convertRawTensor(s))

func bitor_mut*[T](self: var Tensor[T], s: Tensor[T]) =
  ## In-place bitwise `or`.
  rawtensors.bitor_mut(convertRawTensor(self), convertRawTensor(s))

func bitxor_mut*[T](self: var Tensor[T], s: Tensor[T]) =
  ## In-place bitwise `xor`.
  rawtensors.bitxor_mut(convertRawTensor(self), convertRawTensor(s))

func eq*[T](a, b: Tensor[T]): Tensor[T] =
  ## Equality of each tensor values
  convertTensor[T](
    rawtensors.eq(convertRawTensor(a), convertRawTensor(b))
  )
{.pop.}

func `+=`*[T](self: var Tensor[T], b: Tensor[T]) =
  convertRawTensor(self) += convertRawTensor(b)

func `+=`*[T](self: var Tensor[T], s: T) =
  convertRawTensor(self) += s

func `-=`*[T](self: var Tensor[T], b: Tensor[T]) =
  convertRawTensor(self) -= convertRawTensor(b)
func `-=`*[T](self: var Tensor[T], s: T) =
  convertRawTensor(self) -= s

func `*=`*[T](self: var Tensor[T], b: Tensor[T]) =
  convertRawTensor(self) *= convertRawTensor(b)

func `*=`*[T](self: var Tensor[T], s: T) =
  convertRawTensor(self) *= s

func `/=`*[T](self: var Tensor[T], b: Tensor[T]) =
  convertRawTensor(self) /= convertRawTensor(b)

func `/=`*[T](self: var Tensor[T], s: T) =
  convertRawTensor(self) /= s

func equal*[T](a, b: Tensor[T]): bool =
  equal(convertRawTensor(a), convertRawTensor(b))

template `==`*[T](a, b: Tensor[T]): bool =
  a.equal(b)
