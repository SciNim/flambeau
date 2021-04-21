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
  when compileOption("boundChecks"):
    check_index(t, args)
  convertTensor[T](convertRawTensor(t)[args])

template `[]=`*[T](t: var Tensor[T], args: varargs[untyped]): untyped =
  when compileOption("boundChecks"):
    check_index(t, args)
  `[]=`(convertRawTensor(t), args)

# TODO for Arraymancer compatibility
# proc atContiguousIndex*[T](t: Tensor[T], idx: int): T {.noSideEffect,inline.} =
#   ## Return value of tensor at contiguous index
#   ## i.e. as treat the tensor as flattened
#   when T is KnownSupportsCopyMem:
#     return t.unsafe_raw_buf[t.getContiguousIndex(idx)]
#   else:
#     return t.storage.raw_buffer[t.getContiguousIndex(idx)]

# proc atContiguousIndex*[T](t: var Tensor[T], idx: int): var T {.noSideEffect,inline.} =
#   ## Return value of tensor at contiguous index (mutable)
#   ## i.e. as treat the tensor as flattened
#   when T is KnownSupportsCopyMem:
#     return t.unsafe_raw_buf[t.getContiguousIndex(idx)]
#   else:
#     return t.storage.raw_buffer[t.getContiguousIndex(idx)]

# proc atAxisIndex*[T](t: Tensor[T], axis, idx: int, length = 1): Tensor[T] {.noInit,inline.} =
#   ## Returns a sliced tensor in the given axis index

#   when compileOption("boundChecks"):
#     check_axis_index(t, axis, idx, length)

#   result = t
#   result.shape[axis] = length
#   result.offset += result.strides[axis]*idx

# TODO iterators