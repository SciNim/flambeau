import std/[complex, macros]
import cppstl/std_complex
import ../raw/bindings/[rawtensors]
import ../raw/cpp/[std_cpp]
import ../raw/sugar/[indexing]
import ../tensors
import ../raw/sugar/interop as rawinterop

# Checkers func to Raise IndexDefect
# -----------------------------------------------------------------------
func check_index*[T](t: Tensor[T], idx: varargs[int]) {.inline.}=
  if unlikely(idx.len != t.rank):
    raise newException(
      IndexDefect, "Number of arguments: " &
                  $(idx.len) &
                  ", is different from tensor rank: " &
                  $(t.rank)
    )
  for i in 0 ..< t.shape.len:
    if unlikely(not(0 <= idx[i] and idx[i] < t.shape[i])):
      raise newException(
        IndexDefect, "Out-of-bounds access: " &
                    "Tensor of shape " & $t.shape &
                    " being indexed by " & $idx
      )


# Item Access
# -----------------------------------------------------------------------
func item*[T](self: Tensor[T]): T =
  ## Extract the scalar from a 0-dimensional tensor
  result = item(convertRawTensor(self), T)
func item*(self: Tensor[Complex32]): Complex32 =
  item(convertRawTensor(self), typedesc[Complex32]).toCppComplex().toComplex()
func item*(self: Tensor[Complex64]): Complex64 =
  item(convertRawTensor(self), typedesc[Complex64]).toCppComplex().toComplex()

# Indexing
# -----------------------------------------------------------------------
# Unsure what those corresponds to in Python
# func `[]`*[T](self: Tensor, index: Scalar): Tensor
# func `[]`*[T](self: Tensor, index: Tensor): Tensor
# func `[]`*[T](self: Tensor, index: int64): Tensor

func index*[T](self: Tensor[T], args: varargs): Tensor[T] {.noinit.} =
  ## Tensor indexing. It is recommended
  ## to Nimify this in a high-level wrapper.
  ## `tensor.index(indexers)`
  convertTensor[T](
    index(convertRawTensor(self), args)
  )
# We can't use the construct `#.index_put_({@}, #)`
# so hardcode sizes,
# 6d seems reasonable, that would be a batch of 3D videos (videoID/batchID, Time, Color Channel, Height, Width, Depth)
# If you need more you likely aren't indexing individual values.

func index_put*[T](self: var Tensor[T], idx: varargs[int|int64], val: T or Tensor[T]) =
  ## Tensor mutation at index. It is recommended
  convertTensor[T](
    index_put(convertRawTensor(self), idx, val)
  )

# Fancy Indexing
# -----------------------------------------------------------------------
func index_select*[T](self: Tensor[T], axis: int64, indices: Tensor[T]): Tensor[T] {.noinit.} =
  convertTensor[T](
    index_select(convertRawTensor(self), axis, indices)
  )

func masked_select*[T](self: Tensor[T], mask: Tensor[T]): Tensor[T] {.noinit.} =
  convertTensor[T](
    masked_select(convertRawTensor(self), convertRawTensor(mask))
  )

# PyTorch exposes in-place `index_fill_` and `masked_fill_`
# and out-of-place `index_fill` and `masked_fill`
# that does in-place + clone
# we only exposes the in-place version.
func index_fill_mut*[T](self: var Tensor[T], mask: Tensor[T], value: T or Tensor[T]) =
  index_fill_mut(convertRawTensor(self), convertRawTensor(mask), value)

func masked_fill_mut*[T](self: var Tensor[T], mask: Tensor[T], value: T or Tensor[T]) =
  masked_fill_mut(convertRawTensor(self), convertRawTensor(mask), value)

# TODO Move check to func and rewrite those as macros
template `[]`*[T](t: Tensor[T], args: varargs[untyped]): untyped =
  convertTensor[T](convertRawTensor(t)[args])

template `[]=`*[T](t: var Tensor[T], args: varargs[untyped]): untyped =
  `[]=`(convertRawTensor(t), args)

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
proc atContiguousIndex*[T](t: var Tensor[T], idx: int): var T {.noSideEffect,inline.} =
  ## Return value of tensor at contiguous index (mutable)
  ## i.e. as treat the tensor as flattened
  data_ptr(t)[idx]

# TODO Add asContiguous and rowMajor / colMajor utilities

# TODO iterators
