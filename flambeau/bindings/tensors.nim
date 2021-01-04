# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/[strutils, os]

# (Almost) raw bindings to PyTorch Tensors
# -----------------------------------------------------------------------
#
# This provides almost raw bindings to PyTorch tensors.
#
# Differences:
# - `&=`, `|=` and `^=` have been renamed bitand, bitor, bitxor
# - `index` and `index_put` have a common `[]` and `[]=` interface.
#   This allows Nim to be similar to the Python interface.
#   It also avoids exposing the "Slice" and "None" index helpers.
#
# Names were not "Nimified" (camel-cased) to ease
# searching in PyTorch and libtorch docs

# #######################################################################
#
#                          C++ Interop
#
# #######################################################################

# Libraries
# -----------------------------------------------------------------------
# I don't think we can do dynamic loading with C++11
# So link directly

const libPath = currentSourcePath.rsplit(DirSep, 1)[0] & "/../libtorch/lib/"

when defined(windows):
  const libSuffix = ".dll"
elif defined(maxosx): # TODO check this
  const libSuffix = ".dylib" # MacOS
else:
  const libSuffix = ".so" # BSD / Linux

{.link: libPath & "libc10" & libSuffix.}
{.link: libPath & "libtorch_cpu" & libSuffix.}

# Headers
# -----------------------------------------------------------------------

const headersPath = currentSourcePath.rsplit(DirSep, 1)[0] & "/../libtorch/include"
const torchHeadersPath = headersPath / "torch/csrc/api/include"
const torchHeader = torchHeadersPath / "torch/torch.h"

{.passC: "-I" & headersPath.}
{.passC: "-I" & torchHeadersPath.}

{.push header: torchHeader.}

# #######################################################################
#
#                            Tensors
#
# #######################################################################

# TensorOptions
# -----------------------------------------------------------------------
type
  TensorOptions* {.importcpp: "torch::TensorOptions", bycopy.} = object

func init*(T: type TensorOptions): TensorOptions {.constructor,importcpp: "torch::TensorOptions".}

# Scalars
# -----------------------------------------------------------------------
# Scalars are defined in libtorch/include/c10/core/Scalar.h
# as tagged unions of double, int64, complex
# And C++ types are implicitly convertible to Scalar
#
# Hence in Nim we don't need to care about Scalar or defined converters
# (except maybe for complex)
type Scalar* = SomeNumber or bool

# Tensors
# -----------------------------------------------------------------------

type
  Tensor* {.importcpp: "torch::Tensor", byref.} = object

# Strings & Debugging
# -----------------------------------------------------------------------

proc print*(t: Tensor) {.sideeffect, importcpp: "torch::print(@)".}

# Metadata
# -----------------------------------------------------------------------

func dim*(t: Tensor): int64 {.sideeffect, importcpp: "#.dim()".}
func reset*(t: var Tensor) {.importcpp: "#.reset()".}
func `==`*(a, b: Tensor): bool {.importcpp: "#.is_same(#)".}

func ndimension*(t: Tensor): int64 {.importcpp: "#.ndimension()".}
func nbytes*(t: Tensor): uint {.importcpp: "#.nbytes()".}
func numel*(t: Tensor): int64 {.importcpp: "#.numel()".}
func itemsize*(t: Tensor): uint {.importcpp: "#.itemsize()".}
func element_size*(t: Tensor): int64 {.importcpp: "#.element_size()".}

# Backend
# -----------------------------------------------------------------------

func has_storage*(t: Tensor): bool {.importcpp: "#.has_storage()".}
func get_device*(t: Tensor): int64 {.importcpp: "#.get_device()".}
func is_cuda*(t: Tensor): bool {.importcpp: "#.is_cuda()".}
func is_hip*(t: Tensor): bool {.importcpp: "#.is_hip()".}
func is_sparse*(t: Tensor): bool {.importcpp: "#.is_sparse()".}
func is_mkldnn*(t: Tensor): bool {.importcpp: "#.is_mkldnn()".}
func is_vulkan*(t: Tensor): bool {.importcpp: "#.is_vulkan()".}
func is_quantized*(t: Tensor): bool {.importcpp: "#.is_quantized()".}
func is_meta*(t: Tensor): bool {.importcpp: "#.is_meta()".}

# Constructors
# -----------------------------------------------------------------------

func init*(T: type Tensor): Tensor {.constructor,importcpp: "torch::Tensor".}

# Indexing
# -----------------------------------------------------------------------
# libtorch/include/ATen/TensorIndexing.h
# and https://pytorch.org/cppdocs/notes/tensor_indexing.html

# Unsure what those corresponds to in Python
# func `[]`*(a: Tensor, index: Scalar): Tensor {.importcpp: "#[#]".}
# func `[]`*(a: Tensor, index: Tensor): Tensor {.importcpp: "#[#]".}
# func `[]`*(a: Tensor, index: int64): Tensor {.importcpp: "#[#]".}

func index*(a: Tensor): Tensor {.varargs, importcpp: "#.index({@})".}
  ## Tensor indexing. It is recommended
  ## to Nimify this in a high-level wrapper.
  ## `tensor.index(indexers)`

# We can't use the construct `#.index_put_({@}, #)`
# so hardcode sizes,
# 6d seems reasonable, that would be a batch of 3D videos (videoID/batchID, Time, Color Channel, Height, Width, Depth)
# If you need more you likely aren't indexing individual values.

func index_put*(a: var Tensor, i0: auto, val: Scalar or Tensor) {.importcpp: "#.index_put_({#}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(a: var Tensor, i0, i1: auto, val: Scalar or Tensor) {.importcpp: "#.index_put_({#, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(a: var Tensor, i0, i1, i2: auto, val: Scalar or Tensor) {.importcpp: "#.index_put_({#, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(a: var Tensor, i0, i1, i2, i3: auto, val: Scalar or Tensor) {.importcpp: "#.index_put_({#, #, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(a: var Tensor, i0, i1, i2, i3, i4: auto, val: Scalar or Tensor) {.importcpp: "#.index_put_({#, #, #, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(a: var Tensor, i0, i1, i2, i3, i4, i5: auto, val: Scalar or Tensor) {.importcpp: "#.index_put_({#, #, #, #, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.

# Fancy Indexing
# -----------------------------------------------------------------------

func index_select*(a: Tensor, axis: int64, indices: Tensor): Tensor {.importcpp: "#.index_select(@)".}
func masked_select*(a: Tensor, mask: Tensor): Tensor {.importcpp: "#.masked_select(@)".}

# PyTorch exposes in-place `index_fill_` and `masked_fill_`
# and out-of-place `index_fill` and `masked_fill`
# that does in-place + clone
# we only exposes the in-place version.

func index_fill*(a: var Tensor, mask: Tensor, value: Scalar or Tensor) {.importcpp: "#.index_fill_(@)".}
func masked_fill*(a: var Tensor, mask: Tensor, value: Scalar or Tensor) {.importcpp: "#.masked_fill_(@)".}

# High-level indexing API
# -----------------------------------------------------------------------

include ./indexing
macro `[]`*(t: Tensor, args: varargs[untyped]): untyped =
  ## Slice a Tensor
  ## Input:
  ##   - a Tensor
  ##   - and:
  ##     - specific coordinates (``varargs[int]``)
  ##     - or a slice (cf. tutorial)
  ## Returns:
  ##   - a value or a tensor corresponding to the slice
  ##
  ## Usage:
  ##    - Basic indexing - foo[2, 3]
  ##    - Basic indexing - foo[1+1, 2*2*1]
  ##    - Basic slicing - foo[1..2, 3]
  ##    - Basic slicing - foo[1+1..4, 3-2..2]
  ##    - Span slices - foo[_, 3]
  ##    - Span slices - foo[1.._, 3]
  ##    - Span slices - foo[_..3, 3]
  ##    - Span slices - foo[_.._, 3]
  ##    - Stepping - foo[1..3\|2, 3]
  ##    - Span stepping - foo[_.._\|2, 3]
  ##    - Span stepping - foo[_.._\|+2, 3]
  ##    - Span stepping - foo[1.._\|1, 2..3]
  ##    - Span stepping - foo[_..<4\|2, 3]
  ##    - Slicing until at n from the end - foo[0..^4, 3]
  ##    - Span Slicing until at n from the end - foo[_..^2, 3]
  ##    - Stepped Slicing until at n from the end - foo[1..^1\|2, 3]
  ##    - Slice from the end - foo[^1..0\|-1, 3]
  ##    - Slice from the end - expect non-negative step error - foo[^1..0, 3]
  ##    - Slice from the end - foo[^(2*2)..2*2, 3]
  ##    - Slice from the end - foo[^3..^2, 3]
  let new_args = getAST(desugarSlices(args))

  result = quote do:
    slice_typed_dispatch(`t`, `new_args`)

macro `[]=`*(t: var Tensor, args: varargs[untyped]): untyped =
  ## Modifies a tensor inplace at the corresponding location or slice
  ##
  ##
  ## Input:
  ##   - a ``var`` tensor
  ##   - a location:
  ##     - specific coordinates (``varargs[int]``)
  ##     - or a slice (cf. tutorial)
  ##   - a value:
  ##     - a single value that will
  ##       - replace the value at the specific coordinates
  ##       - or be applied to the whole slice
  ##     - an openarray with a shape that matches the slice
  ##     - a tensor with a shape that matches the slice
  ## Result:
  ##   - Nothing, the tensor is modified in-place
  ## Usage:
  ##   - Assign a single value - foo[1..2, 3..4] = 999
  ##   - Assign an array/seq of values - foo[0..1,0..1] = [[111, 222], [333, 444]]
  ##   - Assign values from a view/Tensor - foo[^2..^1,2..4] = bar
  ##   - Assign values from the same Tensor - foo[^2..^1,2..4] = foo[^1..^2|-1, 4..2|-1]

  # varargs[untyped] consumes all arguments so the actual value should be popped
  # https://github.com/nim-lang/Nim/issues/5855

  var tmp = args
  let val = tmp.pop
  let new_args = getAST(desugarSlices(tmp))

  result = quote do:
    slice_typed_dispatch_mut(`t`, `new_args`,`val`)

# Operators
# -----------------------------------------------------------------------

func `not`*(t: Tensor): Tensor {.importcpp: "~#".}
func `-`*(t: Tensor): Tensor {.importcpp: "-#".}
func `+=`*(a: var Tensor, b: Tensor) {.importcpp: "# += #".}
func `+=`*(a: var Tensor, s: Scalar) {.importcpp: "# += #".}
func `-=`*(a: var Tensor, b: Tensor) {.importcpp: "# -= #".}
func `-=`*(a: var Tensor, s: Scalar) {.importcpp: "# -= #".}
func `*=`*(a: var Tensor, b: Tensor) {.importcpp: "# *= #".}
func `*=`*(a: var Tensor, s: Scalar) {.importcpp: "# *= #".}
func `/=`*(a: var Tensor, b: Tensor) {.importcpp: "# /= #".}
func `/=`*(a: var Tensor, s: Scalar) {.importcpp: "# /= #".}
func bitand*(a: var Tensor, s: Tensor) {.importcpp: "# &= #".}
  ## In-place bitwise `and`.
func bitor*(a: var Tensor, s: Tensor) {.importcpp: "# |= #".}
  ## In-place bitwise `or`.
func bitxor*(a: var Tensor, s: Tensor) {.importcpp: "# ^= #".}
  ## In-place bitwise `xor`.

# Functions.h
# -----------------------------------------------------------------------

func eye*(n: int64): Tensor {.importcpp: "torch::eye(@)".}
