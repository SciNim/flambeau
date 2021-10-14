import std/[complex, macros]
import cppstl/std_complex
import ../raw/bindings/[rawtensors]
import ../raw/cpp/[std_cpp]
import ../tensors
import ../raw/sugar/interop as rawinterop

# Checkers func to Raise IndexDefect
# TODO: Where should we use them ?
# -----------------------------------------------------------------------
# func check_index*[T](t: Tensor[T], idx: varargs[int]) {.inline.}=
#   if unlikely(idx.len != t.ndimension):
#     raise newException(
#         IndexDefect, "Error Out-of-bounds access." &
#                     " Index must match Tensor rank ! Expected: " & $(t.ndimension) & ", got: " & $(idx.len) & " elements"
#     )
#   for i in 0 ..< t.shape.len:
#     if unlikely(not(0 <= idx[i] and idx[i] < t.shape[i])):
#       raise newException(
#         IndexDefect, "Error Out-of-bounds access." &
#                     " Index [" & $idx & "] " & " must be in range of Tensor dimensions " & $t.sizes()
#       )

# func check_index*(t: RawTensor, idx: varargs[int]) {.inline.}=
#   if unlikely(idx.len != t.ndimension):
#     raise newException(
#         IndexDefect, "Error Out-of-bounds access." &
#                     " Index must match Tensor rank ! Expected: " & $(t.ndimension) & ", got: " & $(idx.len) & " elements"
#     )
#   for i in 0 ..< t.ndimension:
#     let dim : int64 = t.sizes()[i]
#     if unlikely(not(0 <= idx[i] and idx[i] < dim)):
#       raise newException(
#         IndexDefect, "Error Out-of-bounds access." &
#                     " Index [" & $idx & "] " & " must be in range of Tensor dimensions " & $t.sizes()
#       )

# Item Access
# -----------------------------------------------------------------------
func item*[T](self: Tensor[T]): T =
  when compileOption("boundChecks"):
    if numel(self) > 1:
      raise newException(IndexDefect, ".item() can only be called on one-element Tensor")
  ## Extract the scalar from a 0-dimensional tensor
  result = item(asRaw(self), T)

func item*(self: Tensor[Complex32]): Complex32 =
  when compileOption("boundChecks"):
    if numel(self) > 1:
      raise newException(IndexDefect, ".item() can only be called on one-element Tensor")
  item(asRaw(self), typedesc[Complex32]).toComplex()

func item*(self: Tensor[Complex64]): Complex64 =
  when compileOption("boundChecks"):
    if numel(self) > 1:
      raise newException(IndexDefect, ".item() can only be called on one-element Tensor")
  item(asRaw(self), typedesc[Complex64]).toComplex()

# Indexing
# -----------------------------------------------------------------------
# Unsure what those corresponds to in Python
# func `[]`*[T](self: Tensor, index: Scalar): Tensor
# func `[]`*[T](self: Tensor, index: Tensor): Tensor
# func `[]`*[T](self: Tensor, index: int64): Tensor

template index*[T](self: Tensor[T], args: varargs[untyped]): Tensor[T] =
  ## Tensor indexing
  asTensor[T](
    index(asRaw(self), args)
  )

# Also, this calls rawtensors.index_put which already follow this pattern so writing more code for less bug and more consistency seems worth it
# For reference:
#   We can't use the construct `#.index_put_({@}, #)`
#   so hardcode sizes,
#   6d seems reasonable, that would be a batch of 3D videos (videoID/batchID, Time, Color Channel, Height, Width, Depth)
#   If you need more you likely aren't indexing individual values.
#
# This is kinda ugly but varargs[untyped] + template cause all sort of trouble
func index_put*[T](self: var Tensor[T], i0: auto, val: T or Tensor[T]) =
  when val is Tensor:
    index_put(asRaw(self), i0, asRaw(val))
  elif T is Complex:
    index_put(asRaw(self), i0, toC10_Complex(val))
  else:
    index_put(asRaw(self), i0, val)

func index_put*[T](self: var Tensor[T], i0, i1: auto, val: T or Tensor[T]) =
  when val is Tensor:
    index_put(asRaw(self), i0, i1, asRaw(val))
  elif T is Complex:
    index_put(asRaw(self), i0, i1, toC10_Complex(val))
  else:
    index_put(asRaw(self), i0, i1, val)

func index_put*[T](self: var Tensor[T], i0, i1, i2: auto, val: T or Tensor[T]) =
  when val is Tensor:
    index_put(asRaw(self), i0, i1, i2, asRaw(val))
  elif T is Complex:
    index_put(asRaw(self), i0, i1, i2, toC10_Complex(val))
  else:
    index_put(asRaw(self), i0, i1, i2, val)

func index_put*[T](self: var Tensor[T], i0, i1, i2, i3: auto, val: T or Tensor[T]) =
  when val is Tensor:
    index_put(asRaw(self), i0, i1, i2, i3, asRaw(val))
  elif T is Complex:
    index_put(asRaw(self), i0, i1, i2, i3, toC10_Complex(val))
  else:
    index_put(asRaw(self), i0, i1, i2, i3, val)

func index_put*[T](self: var Tensor[T], i0, i1, i2, i3, i4: auto, val: T or Tensor[T]) =
  when val is Tensor:
    index_put(asRaw(self), i0, i1, i2, i3, i4, asRaw(val))
  elif T is Complex:
    index_put(asRaw(self), i0, i1, i2, i3, i4, toC10_Complex(val))
  else:
    index_put(asRaw(self), i0, i1, i2, i3, i4, val)

func index_put*[T](self: var Tensor[T], i0, i1, i2, i3, i4, i5: auto, val: T or Tensor[T]) =
  when val is Tensor:
    index_put(asRaw(self), i0, i1, i2, i3, i4, i5, asRaw(val))
  elif T is Complex:
    index_put(asRaw(self), i0, i1, i2, i3, i4, i5, toC10_Complex(val))
  else:
    index_put(asRaw(self), i0, i1, i2, i3, i4, i5, val)

# Fancy Indexing
# -----------------------------------------------------------------------
func index_select*[T](self: Tensor[T], axis: int64, indices: Tensor[int64]): Tensor[T] {.noinit.} =
  asTensor[T](
    index_select(asRaw(self), axis, asRaw(indices))
  )

func masked_select*[T](self: Tensor[T], mask: Tensor[int64]): Tensor[T] {.noinit.} =
  asTensor[T](
    masked_select(asRaw(self), asRaw(mask))
  )

# PyTorch exposes in-place `index_fill_` and `masked_fill_`
# and out-of-place `index_fill` and `masked_fill`
# that does in-place + clone
# we only exposes the in-place version.
func index_fill_mut*[T](self: var Tensor[T], mask: Tensor[T], value: T or Tensor[T]) =
  index_fill_mut(asRaw(self), asRaw(mask), value)

func masked_fill_mut*[T](self: var Tensor[T], mask: Tensor[T], value: T or Tensor[T]) =
  masked_fill_mut(asRaw(self), asRaw(mask), value)

# TODO finish porting these Arraymancer proc
# proc getIndex*[T](t: Tensor[T], idx: varargs[int]): int {.noSideEffect,inline.} =
#   ## Convert [i, j, k, l ...] to the proper index.
#   when compileOption("boundChecks"):
#     t.check_index(idx)
#   result = t.offset
#   for i in 0..<idx.len:
#     result += t.strides()[i]*idx[i]

# proc getContiguousIndex*[T](t: Tensor[T], idx: int): int {.noSideEffect,inline.} =
#   result = t.offset
#   if idx != 0:
#     var z = 1
#     for i in countdown(t.rank - 1,0):
#       let coord = (idx div z) mod t.shape[i]
#       result += coord*t.strides[i]
#       z *= t.shape[i]

# proc atIndex*[T](t: Tensor[T], idx: varargs[int]): T {.noSideEffect,inline.} =
#   ## Get the value at input coordinates
#   ## This used to be `[]` before slicing was implemented
#   when T is KnownSupportsCopyMem:
#     result = t.unsafe_raw_buf[t.getIndex(idx)]
#   else:
#     result = t.storage.raw_buffer[t.getIndex(idx)]

# proc atIndex*[T](t: var Tensor[T], idx: varargs[int]): var T {.noSideEffect,inline.} =
#   ## Get the value at input coordinates
#   ## This allows inplace operators t[1,2] += 10 syntax
#   when T is KnownSupportsCopyMem:
#     result = t.unsafe_raw_buf[t.getIndex(idx)]
#   else:
#     result = t.storage.raw_buffer[t.getIndex(idx)]

# proc atIndexMut*[T](t: var Tensor[T], idx: varargs[int], val: T) {.noSideEffect,inline.} =
#   ## Set the value at input coordinates
#   ## This used to be `[]=` before slicing was implemented
#   when T is KnownSupportsCopyMem:
#     t.unsafe_raw_buf[t.getIndex(idx)] = val
#   else:
#     t.storage.raw_buffer[t.getIndex(idx)] = val


# TODO for Arraymancer compatibility
proc atContiguousIndex*[T](t: var Tensor[T], idx: int|int64): var T {.noSideEffect,inline.} =
  ## Return value of tensor at contiguous index (mutable)
  ## i.e. as treat the tensor as flattened
  data_ptr(t)[idx]

# TODO Add asContiguous and rowMajor / colMajor utilities

# TODO iterators
