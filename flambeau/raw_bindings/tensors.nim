# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/[strutils, os, complex],
  ../config,
  ../cpp/std_cpp

# (Almost) raw bindings to PyTorch Tensors
# -----------------------------------------------------------------------
#
# This provides almost raw bindings to PyTorch tensors.
#
# "Nimification" (camelCase), ergonomic indexing and interoperability with Nim types is left to the "high-level" bindings.
# This should ease searching PyTorch and libtorch documentation,
# and make C++ tutorials easily applicable.
#
# Nonetheless some slight modifications were given to the raw bindings:
# - `&=`, `|=` and `^=` have been renamed bitand, bitor, bitxor
# - `[]` and `[]=` are not exported as index and index_put are more flexible
#   and we want to leave those symbols available for Numpy-like ergonomic indexing.
# - Nim's `index_fill_mut` and `masked_fill_mut` are mapped to the in-place
#   C++ `index_fill_` and `masked_fill_`.
#   The original out-of-place versions are doing clone+in-place mutation

# #######################################################################
#
#                          C++ Interop
#
# #######################################################################

# Libraries
# -----------------------------------------------------------------------

const libTorchPath = currentSourcePath.rsplit(DirSep, 1)[0] & "/../../vendor/libtorch"
const librariesPath = libTorchPath & "/lib"

# TODO: we use dynamic linking currently (or are we? unsure about {.link.})
# but we want to provide static linking for dependency-free deployment.
when defined(windows):
  const libSuffix = ".lib"
  const libPrefix = ""
elif defined(maxosx): # TODO check this
  const libSuffix = ".dylib" # MacOS
  const libPrefix = "lib"
else:
  const libSuffix = ".so" # BSD / Linux
  const libPrefix = "lib"

# TODO: proper build system on "nimble install" (put libraries in .nimble/bin?)
# if the libPath is not in LD_LIBRARY_PATH
# The libraries won't be loaded at runtime
when true:
  # Not sure what "link" does differently from standard dynamic linking,
  # it works, it might even work for both GCC and MSVC
  {.link: librariesPath & "/" & libPrefix & "c10" & libSuffix.}
  {.link: librariesPath & "/" & libPrefix & "torch_cpu" & libSuffix.}

  when UseCuda:
    {.link: librariesPath & "/" & libPrefix & "torch_cuda" & libSuffix.}
else:
  # Standard GCC compatible linker
  {.passL: "-L" & librariesPath & " -lc10 -ltorch_cpu ".}

  when UseCuda:
    {.passL: " -ltorch_cuda ".}

# Headers
# -----------------------------------------------------------------------

const headersPath* = libTorchPath & "/include"
const torchHeadersPath* = headersPath / "torch/csrc/api/include"
const torchHeader* = torchHeadersPath / "torch/torch.h"

{.passC: "-I" & headersPath.}
{.passC: "-I" & torchHeadersPath.}

{.push header: torchHeader.}

# Assumptions
# -----------------------------------------------------------------------
#
# LibTorch is using "ArrayRef" through the codebase in particular
# for shapes and strides.
#
# It has the following definition in
# libtorch/include/c10/util/ArrayRef.h
#
# template <typename T>
# class ArrayRef final {
#  private:
#   /// The start of the array, in an external buffer.
#   const T* Data;
#
#   /// The number of elements.
#   size_type Length;
#
# It is noted that the class does not own the underlying data.
# We can model that in a zero-copy and safely borrow-checked way
# with "openarray[T]"

type
  ArrayRef*[T] {.importcpp: "c10::ArrayRef", bycopy.} = object
    # The field are private so we can't use them, but `lent` enforces borrow checking
    p: lent UncheckedArray[T]
    len: csize_t

  IntArrayRef* = ArrayRef[int64]

func data*[T](ar: ArrayRef[T]): ptr UncheckedArray[T] {.importcpp: "#.data()".}
func size*(ar: ArrayRef): csize_t {.importcpp: "#.size()".}

func init*[T](AR: type ArrayRef[T], p: ptr T, len: SomeInteger): ArrayRef[T] {.constructor, importcpp: "c10::ArrayRef<'*0>(@)".}
func init*[T](AR: type ArrayRef[T]): ArrayRef[T] {.constructor, varargs, importcpp: "c10::ArrayRef<'*0>({@})".}

# #######################################################################
#
#                         Tensor Metadata
#
# #######################################################################

# Backend Device
# -----------------------------------------------------------------------
# libtorch/include/c10/core/DeviceType.h
# libtorch/include/c10/core/Device.h

type
  DeviceIndex = int16

  DeviceKind* {.importc: "c10::DeviceType",
                size: sizeof(int16).} = enum
    kCPU = 0
    kCUDA = 1
    kMKLDNN = 2
    kOpenGL = 3
    kOpenCL = 4
    kIDEEP = 5
    kHIP = 6
    kFPGA = 7
    kMSNPU = 8
    kXLA = 9
    kVulkan = 10

  Device* {.importc: "c10::Device", bycopy.} = object
    kind: DeviceKind
    index: DeviceIndex

# Datatypes
# -----------------------------------------------------------------------
# libtorch/include/torch/csrc/api/include/torch/types.h
# libtorch/include/c10/core/ScalarType.h

type
  ScalarKind* {.importc: "torch::ScalarType",
                size: sizeof(int8).} = enum
    kUint8 = 0 # kByte
    kInt8 = 1 # kChar
    kInt16 = 2 # kShort
    kInt32 = 3 # kInt
    kInt64 = 4 # kLong
    kFloat16 = 5 # kHalf
    kFloat32 = 6 # kFloat
    kFloat64 = 7 # kDouble
    kComplexF16 = 8 # kComplexHalf
    kComplexF32 = 9 # kComplexFloat
    kComplexF64 = 10 # kComplexDouble
    kBool = 11
    kQint8 = 12 # Quantized int8
    kQuint8 = 13 # Quantized uint8
    kQint32 = 14 # Quantized int32
    kBfloat16 = 15 # Brain float16


  SomeTorchType* = uint8|byte or SomeSignedInt or
                   SomeFloat or Complex[float32] or Complex[float64]

# TensorOptions
# -----------------------------------------------------------------------
# libtorch/include/c10/core/TensorOptions.h

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

# TensorAccessors
# -----------------------------------------------------------------------
# libtorch/include/ATen/core/TensorAccessors.h
#
# Tensor accessors gives "medium-level" access to a Tensor raw-data
# - Compared to low-level "data_ptr" they take care of striding and shape
# - Compared to high-level functions they don't provide any parallelism.

# #######################################################################
#
#                            Tensors
#
# #######################################################################

# Tensors
# -----------------------------------------------------------------------

type
  Tensor* {.importcpp: "torch::Tensor", bycopy.} = object

{.push header: "<tuple>".}
type
  CppTuple2Tensors* {.importcpp: "std::tuple<Tensor, Tensor>", bycopy.} = object

func getFirst*(t: CppTuple2Tensors): lent Tensor {.importcpp: "std::get<0>(#)".}
func getSecond*(t: CppTuple2Tensors): lent Tensor {.importcpp: "std::get<1>(#)".}


# Strings & Debugging
# -----------------------------------------------------------------------

proc print*(t: Tensor) {.sideeffect, importcpp: "torch::print(@)".}

# Metadata
# -----------------------------------------------------------------------

func dim*(t: Tensor): int64 {.importcpp: "#.dim()".}
func reset*(t: var Tensor) {.importcpp: "#.reset()".}
func `==`*(a, b: Tensor): bool {.importcpp: "#.is_same(#)".}

func sizes*(a: Tensor): IntArrayRef {.importcpp:"#.sizes()".}
  ## This is Arraymancer and Numpy "shape"
func strides*(a: Tensor): IntArrayRef {.importcpp:"#.strides()".}

func ndimension*(t: Tensor): int64 {.importcpp: "#.ndimension()".}
func nbytes*(t: Tensor): uint {.importcpp: "#.nbytes()".}
func numel*(t: Tensor): int64 {.importcpp: "#.numel()".}
  ## This is Arraymancer and Numpy "size"
func itemsize*(t: Tensor): uint {.importcpp: "#.itemsize()".}
func element_size*(t: Tensor): int64 {.importcpp: "#.element_size()".}

# Accessors
# -----------------------------------------------------------------------

func data_ptr*(t: Tensor, T: typedesc[SomeTorchType]): ptr UncheckedArray[T] {.importcpp: "#.data_ptr<'2>(#)".}
  ## Gives raw access to a tensor data of type T.
  ##
  ## This is a very low-level procedure. You need to take care
  ## of the tensor shape and strides yourself.
  ##
  ## It is recommended to use this only on contiguous tensors
  ## (freshly created or freshly cloned) and to avoid
  ## sliced tensors.

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

# DeviceType and ScalarType are auto-convertible to TensorOptions

func init*(T: type Tensor): Tensor {.constructor,importcpp: "torch::Tensor".}

func from_blob*(data: pointer, sizes: IntArrayRef, options: TensorOptions): Tensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: IntArrayRef, scalarKind: ScalarKind): Tensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: IntArrayRef, device: DeviceKind): Tensor {.importcpp: "torch::from_blob(@)".}

func from_blob*(data: pointer, sizes: int64, options: TensorOptions): Tensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: int64, scalarKind: ScalarKind): Tensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: int64, device: DeviceKind): Tensor {.importcpp: "torch::from_blob(@)".}

func from_blob*(data: pointer, sizes, strides: IntArrayRef, options: TensorOptions): Tensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes, strides: IntArrayRef, scalarKind: ScalarKind): Tensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes, strides: IntArrayRef, device: DeviceKind): Tensor {.importcpp: "torch::from_blob(@)".}

func empty*(size: IntArrayRef, options: TensorOptions): Tensor {.importcpp:"torch::empty(@)"}
  ## Create an uninitialized tensor of shape `size`
  ## The tensor data must be filled manually
  ##
  ## The output tensor will be row major (C contiguous)
func empty*(size: IntArrayRef, scalarKind: ScalarKind): Tensor {.importcpp:"torch::empty(@)"}
func empty*(size: IntArrayRef, device: DeviceKind): Tensor {.importcpp:"torch::empty(@)"}
  ## Create an uninitialized tensor of shape `size`
  ## The tensor data must be filled manually.
  ##
  ## If device is NOT on CPU make sure to use specialized
  ## copy operations. For example to update on Cuda devices
  ## use cudaMemcpy not a.data[i] = 123
  ##
  ## The output tensor will be row major (C contiguous)

func clone*(a: Tensor): Tensor {.importcpp: "#.clone()".}

func cpu*(a: Tensor): Tensor {.importcpp: "#.cpu()".}
func cuda*(a: Tensor): Tensor {.importcpp: "#.cuda()".}
func hip*(a: Tensor): Tensor {.importcpp: "#.hip()".}
func vulkan*(a: Tensor): Tensor {.importcpp: "#.vulkan()".}

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

func index_fill_mut*(a: var Tensor, mask: Tensor, value: Scalar or Tensor) {.importcpp: "#.index_fill_(@)".}
func masked_fill_mut*(a: var Tensor, mask: Tensor, value: Scalar or Tensor) {.importcpp: "#.masked_fill_(@)".}

# Low-level slicing API
# -----------------------------------------------------------------------

type
  TorchSlice* {.importcpp: "torch::indexing::Slice", bycopy.} = object
  # libtorch/include/ATen/TensorIndexing.h

  IndexNone* {.importcpp: "torch::indexing::None", bycopy.} = object
    ## enum class TensorIndexType
  IndexEllipsis* {.importcpp: "torch::indexing::Ellipsis", bycopy.} = object
    ## enum class TensorIndexType

  Ellipsis* = IndexEllipsis
  SomeSlicer* = IndexNone or SomeSignedInt

func torchSlice*(){.importcpp: "torch::indexing::Slice(@)", constructor.}
func torchSlice*(start: SomeSlicer): TorchSlice {.importcpp: "torch::indexing::Slice(@)", constructor.}
func torchSlice*(start: SomeSlicer, stop: SomeSlicer): TorchSlice {.importcpp: "torch::indexing::Slice(@)", constructor.}
func torchSlice*(start: SomeSlicer, stop: SomeSlicer, step: SomeSlicer): TorchSlice {.importcpp: "torch::indexing::Slice(@)", constructor.}
func start*(s: TorchSlice): int64 {.importcpp: "#.start()".}
func stop*(s: TorchSlice): int64 {.importcpp: "#.stop()".}
func step*(s: TorchSlice): int64 {.importcpp: "#.step()".}

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

func toType*(t: Tensor, dtype: ScalarKind): Tensor {.importcpp: "#.toType(@)".}

func eye*(n: int64): Tensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, options: TensorOptions): Tensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, scalarKind: ScalarKind): Tensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, device: DeviceKind): Tensor {.importcpp: "torch::eye(@)".}


func add*(t: Tensor, other: Tensor, alpha: Scalar = 1): Tensor {.importcpp: "#.add(@)".}
func add*(t: Tensor, other: Scalar, alpha: Scalar = 1): Tensor {.importcpp: "#.add(@)".}
func addmv*(t: Tensor, mat: Tensor, vec: Tensor, beta: Scalar = 1, alpha: Scalar = 1): Tensor {.importcpp: "#.addmv(@)".}
func addmm*(t, mat1, mat2: Tensor, beta: Scalar = 1, alpha: Scalar = 1): Tensor {.importcpp: "#.addmm(@)".}
func mm*(t, other: Tensor): Tensor {.importcpp: "#.mm(@)".}
func matmul*(t, other: Tensor): Tensor {.importcpp: "#.matmul(@)".}
func bmm*(t, other: Tensor): Tensor {.importcpp: "#.bmm(@)".}

func luSolve*(t, data, pivots: Tensor): Tensor {.importcpp: "#.lu_solve(@)".}

func qr_internal*(t: Tensor, some: bool = true): lent CppTuple2Tensors {.importcpp: "#.qr(@)".}
func qr*(t: Tensor, some: bool = true): (lent Tensor, lent Tensor) =
  let cppTuple = qr_internal(t, some)
  result[0] = cppTuple.getFirst()
  result[1] = cppTuple.getSecond()

# addr?
func all*(t: Tensor, axis: int64): Tensor {.importcpp: "#.all(@)".}
func all*(t: Tensor, axis: int64, keepdim: bool): Tensor {.importcpp: "#.all(@)".}
func allClose*(t, other: Tensor, rtol: float64 = 1e-5, abstol: float64 = 1e-8, equalNan: bool = false): bool {.importcpp: "#.allclose(@)".}
func any*(t: Tensor, axis: int64): Tensor {.importcpp: "#.any(@)".}
func any*(t: Tensor, axis: int64, keepdim: bool): Tensor {.importcpp: "#.any(@)".}
func argmax*(t: Tensor): Tensor {.importcpp: "#.argmax()".}
func argmax*(t: Tensor, axis: int64, keepdim: bool = false): Tensor {.importcpp: "#.argmax(@)".}
func argmin*(t: Tensor): Tensor {.importcpp: "#.argmin()".}
func argmin*(t: Tensor, axis: int64, keepdim: bool = false): Tensor {.importcpp: "#.argmin(@)".}

# aggregate:

# sum needs wrapper procs/templates to allow for using nim arrays and single axis.
func sum*(t: Tensor): Tensor {.importcpp: "#.sum()".}
func sum*(t: Tensor, dtype: ScalarKind): Tensor {.importcpp: "#.sum(@)".}
func sum*(t: Tensor, axis: int64, keepdim: bool = false): Tensor {.importcpp: "#.sum(@)".}
func sum*(t: Tensor, axis: int64, keepdim: bool = false, dtype: ScalarKind): Tensor {.importcpp: "#.sum(@)".}
func sum*(t: Tensor, axis: IntArrayRef, keepdim: bool = false): Tensor {.importcpp: "#.sum(@)".}
func sum*(t: Tensor, axis: IntArrayRef, keepdim: bool = false, dtype: ScalarKind): Tensor {.importcpp: "#.sum(@)".}

# mean as well
func mean*(t: Tensor): Tensor {.importcpp: "#.mean()".}
func mean*(t: Tensor, dtype: ScalarKind): Tensor {.importcpp: "#.mean(@)".}
func mean*(t: Tensor, axis: int64, keepdim: bool = false): Tensor {.importcpp: "#.mean(@)".}
func mean*(t: Tensor, axis: int64, keepdim: bool = false, dtype: ScalarKind): Tensor {.importcpp: "#.mean(@)".}
func mean*(t: Tensor, axis: IntArrayRef, keepdim: bool = false): Tensor {.importcpp: "#.mean(@)".}
func mean*(t: Tensor, axis: IntArrayRef, keepdim: bool = false, dtype: ScalarKind): Tensor {.importcpp: "#.mean(@)".}

# median requires std::tuple

func prod*(t: Tensor): Tensor {.importcpp: "#.prod()".}
func prod*(t: Tensor, dtype: ScalarKind): Tensor {.importcpp: "#.prod(@)".}
func prod*(t: Tensor, axis: int64, keepdim: bool = false): Tensor {.importcpp: "#.prod(@)".}
func prod*(t: Tensor, axis: int64, keepdim: bool = false, dtype: ScalarKind): Tensor {.importcpp: "#.prod(@)".}

func min*(t: Tensor): Tensor {.importcpp: "#.min()".}
# Must wrap CppTuple
func min*(t: Tensor, axis: int64, keepdim: bool = false): CppTuple2Tensors {.importcpp: "torch::min(@)".}
func max*(t: Tensor): Tensor {.importcpp: "#.max()".}
# Must wrap CppTuple
#func max*(t: Tensor, axis: int64, keepdim: bool = false): CppTuple[Tensor, Tensor] {.importcpp: "torch::max(@)".}

func variance*(t: Tensor, unbiased: bool = true): Tensor {.importcpp: "#.var(@)".} # can't use `var` because of keyword.
func variance*(t: Tensor, axis: int64, unbiased: bool = true, keepdim: bool = false): Tensor {.importcpp: "#.var(@)".}
func variance*(t: Tensor, axis: IntArrayRef, unbiased: bool = true, keepdim: bool = false): Tensor {.importcpp: "#.var(@)".}

func stddev*(t: Tensor, unbiased: bool = true): Tensor {.importcpp: "#.std(@)".}
func stddev*(t: Tensor, axis: int64, unbiased: bool = true, keepdim: bool = false): Tensor {.importcpp: "#.std(@)".}
func stddev*(t: Tensor, axis: IntArrayRef, unbiased: bool = true, keepdim: bool = false): Tensor {.importcpp: "#.std(@)".}

# algorithms:

#func sort*(t: Tensor, axis: int64 = -1, descending: bool = false): CppTuple[Tensor, Tensor] {.importcpp: "#.sort(@)".}
func argsort*(t: Tensor, axis: int64 = -1, descending: bool = false): Tensor {.importcpp: "#.argsort(@)".}

# math
func abs*(t: Tensor): Tensor {.importcpp: "#.abs()".}
func absolute*(t: Tensor): Tensor {.importcpp: "#.absolute()".}
func angle*(t: Tensor): Tensor {.importcpp: "#.angle()".}
func sgn*(t: Tensor): Tensor {.importcpp: "#.sgn()".}
func conj*(t: Tensor): Tensor {.importcpp: "#.conj()".}
func acos*(t: Tensor): Tensor {.importcpp: "#.acos()".}
func arccos*(t: Tensor): Tensor {.importcpp: "#.arccos()".}
func acosh*(t: Tensor): Tensor {.importcpp: "#.acosh()".}
func arccosh*(t: Tensor): Tensor {.importcpp: "#.arccosh()".}
func asinh*(t: Tensor): Tensor {.importcpp: "#.asinh()".}
func arcsinh*(t: Tensor): Tensor {.importcpp: "#.arcsinh()".}
func atanh*(t: Tensor): Tensor {.importcpp: "#.atanh()".}
func arctanh*(t: Tensor): Tensor {.importcpp: "#.arctanh()".}
func asin*(t: Tensor): Tensor {.importcpp: "#.asin()".}
func arcsin*(t: Tensor): Tensor {.importcpp: "#.arcsin()".}
func atan*(t: Tensor): Tensor {.importcpp: "#.atan()".}
func arctan*(t: Tensor): Tensor {.importcpp: "#.arctan()".}
func cos*(t: Tensor): Tensor {.importcpp: "#.cos()".}
func sin*(t: Tensor): Tensor {.importcpp: "#.sin()".}
func tan*(t: Tensor): Tensor {.importcpp: "#.tan()".}
func exp*(t: Tensor): Tensor {.importcpp: "#.exp()".}
func exp2*(t: Tensor): Tensor {.importcpp: "#.exp2()".}
func erf*(t: Tensor): Tensor {.importcpp: "#.erf()".}
func erfc*(t: Tensor): Tensor {.importcpp: "#.erfc()".}
func reciprocal*(t: Tensor): Tensor {.importcpp: "#.reciprocal()"}
func neg*(t: Tensor): Tensor {.importcpp: "#.neg()".}
func clamp*(t: Tensor, min, max: Scalar): Tensor {.importcpp: "#.clamp(@)".}
func clampMin*(t: Tensor, min: Scalar): Tensor {.importcpp: "#.clamp_min(@)".}
func clampMax*(t: Tensor, max: Scalar): Tensor {.importcpp: "#.clamp_max(@)".}

func dot*(t: Tensor, other: Tensor): Tensor {.importcpp: "#.dot(@)".}

func squeeze*(t: Tensor): Tensor {.importcpp: "#.squeeze()".}
func squeeze*(t: Tensor, axis: int64): Tensor {.importcpp: "#.squeeze(@)".}
func unsqueeze*(t: Tensor, axis: int64): Tensor {.importcpp: "#.unsqueeze(@)".}

func fft*(t: Tensor): Tensor {.importcpp: "torch::fft_fft(@)".}
func fft*(t: Tensor, n: int64, axis: int64 = -1): Tensor {.importcpp: "torch::fft_fft(@)".}
func fft*(t: Tensor, n: int64, axis: int64 = -1, norm: CppString): Tensor {.importcpp: "torch::fft_fft(@)".}

#func convolution*(t: Tensor, weight: Tensor, bias: Tensor, stride, padding, dilation: int64, transposed: bool, outputPadding: int64, groups: int64): Tensor {.importcpp: "torch::convolution(@)".}
