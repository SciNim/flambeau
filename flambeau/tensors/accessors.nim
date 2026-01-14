import std/[complex, macros]
import cppstl/std_complex

import ../raw/bindings/[rawtensors, c10]
import ../raw/cpp/[std_cpp]
import ../tensors
import ../raw/sugar/rawinterop

let t_dont_use_this {.used.} = initRawTensor()

# Bounds checking functions
# -----------------------------------------------------------------------
func check_index*[T](t: Tensor[T], idx: varargs[int]) {.inline.} =
  ## Check tensor indexing bounds (delegates to RawTensor version)
  check_index(asRaw(t), idx)

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
    if numel(asRaw(self)) > 1:
      raise newException(IndexDefect, ".item() can only be called on one-element Tensor")
  item(asRaw(self), typedesc[Complex32]).toComplex()

func item*(self: Tensor[Complex64]): Complex64 =
  when compileOption("boundChecks"):
    if numel(asRaw(self)) > 1:
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
  asTensor[T](index(asRaw(self), args))

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
func index_select*[T](self: Tensor[T], axis: int64, indices: Tensor[int64]): Tensor[T] =
  asTensor[T](index_select(asRaw(self), axis, asRaw(indices)))

func masked_select*[T](self: Tensor[T], mask: Tensor[int64]): Tensor[T] =
  asTensor[T](masked_select(asRaw(self), asRaw(mask)))

# PyTorch exposes in-place `index_fill_` and `masked_fill_`
# and out-of-place `index_fill` and `masked_fill`
# that does in-place + clone
# we only exposes the in-place version.
func index_fill_mut*[T](self: var Tensor[T], mask: Tensor[T], value: T or Tensor[T]) =
  index_fill_mut(asRaw(self), asRaw(mask), value)

func masked_fill_mut*[T](self: var Tensor[T], mask: Tensor[T], value: T or Tensor[T]) =
  masked_fill_mut(asRaw(self), asRaw(mask), value)

# Arraymancer-compatible indexing utilities
# -----------------------------------------------------------------------

func getIndex*[T](t: Tensor[T], idx: varargs[int]): int {.inline.} =
  ## Convert [i, j, k, l ...] to the proper linear index.
  ## This computes the flattened index from multi-dimensional coordinates.
  when compileOption("boundChecks"):
    # Inline bounds checking to avoid C++ linkage issues
    let ndim = t.ndimension
    if unlikely(idx.len != ndim):
      raise newException(
        IndexDefect,
        "Error Out-of-bounds access. Index must match Tensor rank! Expected: " & $ndim & ", got: " & $(idx.len),
      )
    let sizes = t.sizes()
    for i in 0 ..< idx.len:
      let dim_size: int64 = sizes[i]
      if unlikely(not (0 <= idx[i] and idx[i] < dim_size)):
        raise newException(IndexDefect, "Error Out-of-bounds access. Index must be in valid range.")

  result = 0
  let strides = t.strides()
  for i in 0 ..< idx.len:
    result += int(strides[i]) * idx[i]

func getContiguousIndex*[T](t: Tensor[T], idx: int): int {.inline.} =
  ## Get the storage index from a contiguous (flattened) index.
  ## This converts a linear index back to the actual storage location,
  ## accounting for strides.
  result = 0
  if idx != 0:
    var z = 1
    let sizes = t.sizes()
    let strides = t.strides()
    for i in countdown(t.ndimension - 1, 0):
      let shape_i = int(sizes[i])
      let coord = (idx div z) mod shape_i
      result += coord * int(strides[i])
      z *= shape_i

func atIndex*[T](t: Tensor[T], idx: varargs[int]): T {.inline.} =
  ## Get the value at input coordinates.
  ## This is the immutable accessor that returns a copy of the value.
  ## For tensors, prefer using `t[i, j, k].item()` which returns a Tensor.
  let linear_idx = t.getIndex(idx)
  let p = data_ptr(t)
  result = p[linear_idx]

proc atIndex*[T](t: var Tensor[T], idx: varargs[int]): var T {.inline.} =
  ## Get mutable reference to value at input coordinates.
  ## This allows in-place operators like `t.atIndex(1, 2) += 10`.
  ##
  ## Note: For the more ergonomic `t[1, 2] += 10` syntax,
  ## use the `indexedMutate` macro instead.
  let linear_idx = t.getIndex(idx)
  let p = data_ptr(t)
  result = p[linear_idx]

proc atIndexMut*[T](t: var Tensor[T], idx: varargs[int], val: T) {.inline.} =
  ## Set the value at input coordinates.
  ## This is the mutable setter for direct value assignment.
  let linear_idx = t.getIndex(idx)
  let p = data_ptr(t)
  p[linear_idx] = val

func atContiguousIndex*[T](t: Tensor[T], idx: int | int64): T {.inline.} =
  ## Return value of tensor at contiguous index (immutable).
  ## Treats the tensor as flattened.
  let p = data_ptr(t)
  result = p[idx]

proc atContiguousIndex*[T](t: var Tensor[T], idx: int | int64): var T {.inline.} =
  ## Return value of tensor at contiguous index (mutable).
  ## Treats the tensor as flattened.
  ## This allows in-place operations on flattened indices.
  let p = data_ptr(t)
  result = p[idx]

# TODO Add asContiguous and rowMajor / colMajor utilities

# TODO iterators
