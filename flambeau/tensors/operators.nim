import ../raw/bindings/[rawtensors, c10]
import ../raw/sugar/[interop, indexing]
import ../tensors
import std/[complex, macros]

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

func `-`*[T](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  asTensor[T](asRaw(a) - asRaw(b))

func `*`*[T](a: Tensor[T], b: Tensor[T]): Tensor[T] =
  asTensor[T](asRaw(a) * asRaw(b))

func `*`*[T](a: SomeNumber, b: Tensor[T]): Tensor[T] =
  asTensor[T](a.cdouble * asRaw(b))

func `*`*[T](a: Tensor[T], b: SomeNumber): Tensor[T] =
  asTensor[T](asRaw(a) * b.cdouble)

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
  asTensor[T](
    rawtensors.eq(asRaw(a), asRaw(b))
  )
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
