# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  ../cpp/std_cpp,
  ../libtorch,
  ./tensors,
  ./c10

# (Almost) raw bindings to PyTorch Neural Networks
# -----------------------------------------------------------------------
#
# This provides almost raw bindings to PyTorch tensors.
#
# "Nimification" (camelCase), ergonomic indexing and interoperability with Nim types is left to the "high-level" bindings.
# This should ease searching PyTorch and libtorch documentation,
# and make C++ tutorials easily applicable.

# C++ interop
# -----------------------------------------------------------------------

{.push cdecl.}
{.push header: torchHeader.}

# #######################################################################
#
#                       Autograd
#
# #######################################################################

type
  AutoGradMode* {.bycopy, pure, inheritable, importcpp: "torch::AutoGradMode".} = object

func autogradMode(enabled: bool): AutoGradMode {.constructor, importcpp: "torch::AutoGradMode(#)".}

template with*(T: type AutoGradMode, enabled: bool, body: untyped): untyped =
  bind autogradMode
  block:
    let gradMode = autogradMode(enabled)
    body

template no_grad_mode*(body: untyped): untyped =
  ## Disable precomputations necessary for gradient propagation
  with(AutoGradMode, enabled = false):
    body

# #######################################################################
#
#                       LibTorch Functional API
#
# #######################################################################
#
# LibTorch Functional API is described here.
# https://pytorch.org/cppdocs/api/namespace_torch__nn__functional.html#namespace-torch-nn-functional
# libtorch/include/torch/csrc/api/include/torch/nn/functional
#
# It is stateless, meaning users need to track weight and bias themselves.
# It is suitable for layers with no learning parameters (for example reshaping),
# or when extra flexibility is required at a small price of ergonomics.
# The high-level Module API uses Functional internally.
#
# Note:
#   Function exists both in ATen TensorBody.h (namespace at:: or torch::)
#   and in torch::nn::functional.
#
#   We can have
#     func dropout*(input: Tensor, p = 0.5, training=true): Tensor {.importcpp: "torch::nn::functional::dropout(@)".}
#     func dropout_mut*(input: var Tensor, p = 0.5, training=true) {.importcpp: "torch::nn::functional::dropout(@, /*inplace=*/ true)".}
#
#     OR
#
#     func dropout*(input: Tensor, p = 0.5, training=true): Tensor {.importcpp: "torch::dropout(@)".}
#     func dropout_mut*(input: var Tensor, p = 0.5, training=true) {.importcpp: "torch::dropout_(@)".}
#
#   The functions in torch::nn::functional are thin inlined wrapper over TensorBody.h
#   so we directly use them.

# Linear Layers
# -------------------------------------------------------------------------

func linear*(input, weight: Tensor): Tensor {.importcpp: "torch::nn::functional::linear(@)".}
  ## Applies a linear transformation to the incoming data:
  ##   y = input * transpose(weight)
  ##
  ## Input: (N,∗,in_features)(N, *, in\_features)(N,∗,in_features)
  ##        N is the batch size, * means any number of additional dimensions
  ## Weight: (out_features,in_features)
  ## Output: (N,∗,out_features)

func linear*(input, weight, bias: Tensor): Tensor {.importcpp: "torch::nn::functional::linear(@)".}
  ## Applies a linear transformation to the incoming data:
  ##   y = input * transpose(weight) + bias
  ##
  ## Input: (N,∗,in_features)(N, *, in\_features)(N,∗,in_features)
  ##        N is the batch size, * means any number of additional dimensions
  ## Weight: (out_features,in_features)
  ## Bias: (out_features)
  ## Output: (N,∗,out_features)

# Pooling functions
# -------------------------------------------------------------------------

func max_pool2d*(input: Tensor): Tensor {.varargs, importcpp:"torch::max_pool2d(#, {@})".}
  ## MaxPool 2D function
  ## - `input`: a Tensor
  ## - `kernel_size`: the kernel shape

func max_pool2d*(input: Tensor, kernel_size: IntArrayRef): Tensor {.importcpp:"torch::max_pool2d(@)".}

# Activation functions
# -------------------------------------------------------------------------

func relu*(input: Tensor): Tensor {.importcpp: "torch::relu(@)".}
func relu_mut*(input: var Tensor) {.importcpp: "torch::relu_(@)".}

func log_softmax*(input: Tensor, axis: int64): Tensor {.importcpp: "torch::log_softmax(@)".}
func log_softmax*(input: Tensor, axis: int64, dtype: ScalarKind): Tensor {.importcpp: "torch::log_softmax(@)".}

# Dropout functions
# -------------------------------------------------------------------------

func dropout*(input: Tensor, p = 0.5, training=true): Tensor {.importcpp: "torch::dropout(@)".}
func dropout_mut*(input: var Tensor, p = 0.5, training=true) {.importcpp: "torch::dropout_(@)".}

# Loss functions
# -------------------------------------------------------------------------

type
  Reduction* {.size: sizeof(cint), importcpp:"torch::Reduction::Reduction".} = enum
    None = 0 # Do not reduce
    Mean = 1 # (Possibly weighted) mean of losses
    Sum = 2  # Sum losses

func nll_loss*(input, target: Tensor): Tensor {.importcpp: "torch::nll_loss(@)".}
func nll_loss*(input, target: Tensor, red: Reduction): Tensor {.importcpp: "torch::nll_loss(#, #, /*weight=*/{}, #)".}

func binary_cross_entropy_with_logits*(input, target: Tensor): Tensor {.importcpp: "torch::binary_cross_entropy_with_logits(@)".}
  ## Sigmoid + Log + Negative loglikelihood
  ## PyTorch naming
func sigmoid_cross_entropy*(input, target: Tensor): Tensor {.importcpp: "torch::binary_cross_entropy_with_logits(@)".}
  ## Sigmoid + Log + Negative loglikelihood
  ## Arraymancer or Tensorflow naming

# #######################################################################
#
#                       LibTorch Module API
#
# #######################################################################
#
# LibTorch Module API is described here.
# https://pytorch.org/cppdocs/api/namespace_torch__nn.html#classes
# libtorch/include/torch/csrc/api/include/torch/nn/module.h
#
# It uses class derived from the base "Module" class.
# The modules keep track of weights and biases for the users.
# They also keep track of the training or evaluation mode,
# allow pretty-printing of a computation graph,
# serialization and deserialization.
#
# See Module ownership notes:
# - https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module_holder.html#classtorch_1_1nn_1_1_module_holder
# - https://pytorch.org/tutorials/advanced/cpp_frontend.html#module-ownership
# all modules are thin wrapper around shared_ptr + ModuleImpl
#
# Torch serialization expect the shared_ptr so we should respect their Module API.

type
  Module* {.bycopy, pure, inheritable, importcpp: "torch::nn::Module".} = object
    ## A LibTorch neural network module that can be inherited from
    # Impl detaim:
    #   Nim inheritable objects have runtime type information pointer
    #   as a hidden first field.
    #   {.pure, inheritable.} removes that to make the object C++ compatible.
  ModuleHolder* {.bycopy, pure, inheritable, importcpp: "torch::nn::ModuleHolder".} = object

  SharedModule*[T: Module] = CppSharedPtr[T]

proc register_module*[ParMod: ModuleHolder, ChildMod: ModuleHolder](
       parent: var ParMod, name: cstring, child: var ChildMod)
       {.importcpp: "#.register_module(@)".}
  ## Register a submodule to a parent module.

proc register_module*[ParMod: ModuleHolder, ChildMod: ModuleHolder](
       parent: var ParMod, name: cstring, child: sink ChildMod): ChildMod
       {.importcpp: "#.register_module(@)".}
  ## Register a submodule to a parent module.

proc register_module*[ParMod: Module, ChildMod: ModuleHolder](
       parent: var SharedModule[ParMod], name: cstring, child: var ChildMod)
       {.importcpp: "#->register_module(@)".}
  ## Register a submodule to a parent module.

proc register_module*[ParMod: Module, ChildMod: ModuleHolder](
       parent: var SharedModule[ParMod], name: cstring, child: sink ChildMod): ChildMod
       {.importcpp: "#->register_module(@)".}
  ## Register a submodule to a parent module.

proc register_module*[ParMod: Module, ChildMod: Module](
       parent: var SharedModule[ParMod], name: cstring, child: var SharedModule[ChildMod])
       {.importcpp: "#->register_module(@)".}
  ## Register a submodule to a parent module.

proc register_module*[ParMod: Module, ChildMod: Module](
       parent: var SharedModule[ParMod], name: cstring, child: sink SharedModule[ChildMod]): SharedModule[ChildMod]
       {.importcpp: "#->register_module(@)".}
  ## Register a submodule to a parent module.

proc register_parameter*[ParMod: Module](
       parent: var SharedModule[ParMod], name: cstring, child: sink Tensor): Tensor
       {.importcpp: "#->register_parameter(@)".}
  ## Register a submodule to a parent module.

func parameters*(module: Module, recurse = true): CppVector[Tensor]{.importcpp: "#.parameters(#)".}

func is_training*(module: Module): bool {.importcpp: "#.is_training()".}

proc to*(module: ModuleHolder or SharedModule, device: DeviceKind) {.importcpp: "#->to(#)".}
proc to*(module: ModuleHolder or SharedModule, device: Device) {.importcpp: "#->to(#)".}

func train*(module: var ModuleHolder or SharedModule, on = true) {.importcpp: "#->train(#)".}
  ## Enable training mode

func eval*(module: var Module or SharedModule) {.importcpp: "#->eval()".}
  ## Enable evaluation mode

# Linear layer
# --------------------------------
# https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_linear_impl.html

type
  LinearOptions* {.bycopy, importcpp: "torch::nn::LinearOptions".} = object

  Linear* {.pure, bycopy, importcpp: "torch::nn::Linear".} = object of ModuleHolder
    # Linear is a shared_ptr underneath.
    # The ptr is bycopy which results in the actual data being byref.
    options*{.importc.}: LinearOptions
    weight*{.importc.}: Tensor
    bias*{.importc.}: Tensor

func init*(T: type Linear, in_features, out_features: int64): T {.constructor, importcpp:"torch::nn::Linear(@)".}

func reset*(linear: Linear){.importcpp: "#.reset()".}
  ## reset() must perform initialization of all members with reference semantics,
  ## most importantly parameters, buffers and submodules.

func reset_parameters*(linear: Linear){.importcpp: "#.reset_parameters()".}

# pretty_print

func forward*(linear: Linear, input: Tensor): Tensor {.importcpp: "#->forward(#)".}
  ## Transforms the ``input`` tensor
  ## by multiplying with the ``weight``
  ## and optionally adding the ``bias``,
  ## if ``with_bias`` is true in the ``options``.

# Conv2D layer
# --------------------------------
# Link TODO

type
  Conv2dOptions* {.bycopy, importcpp: "torch::nn::Conv2dOptions".} = object

  Conv2d* {.pure, bycopy, importcpp: "torch::nn::Conv2d".} = object of ModuleHolder
    # Conv2d is a shared_ptr underneath.
    # The ptr is bycopy which results in the actual data being byref.
    options*{.importc.}: Conv2DOptions
    weight*{.importc.}: Tensor
    bias*{.importc.}: Tensor

func init*(T: type Conv2d, in_channels, out_channels, kernel_size: int64): T {.constructor, importcpp:"torch::nn::Conv2d(@)".}
func init*(T: type Conv2d, in_channels, out_channels,
           kernel_size: array[2, int64]): T {.constructor, importcpp:"torch::nn::Conv2d(@)".}

func reset*(conv2d: Conv2d){.importcpp: "#.reset()".}
  ## reset() must perform initialization of all members with reference semantics,
  ## most importantly parameters, buffers and submodules.

func reset_parameters*(conv2d: Conv2d){.importcpp: "#.reset_parameters()".}

# pretty_print

func forward*(conv2d: Conv2d, input: Tensor): Tensor {.importcpp: "#->forward(#)".}
  ## Transforms the ``input`` tensor
  ## by multiplying with the ``weight``
  ## and optionally adding the ``bias``,
  ## if ``with_bias`` is true in the ``options``.

# Dropout layers
# --------------------------------
# Link TODO

type
  DropoutOptions* {.bycopy, importcpp: "torch::nn::DropoutOptions".} = object

  Dropout* {.pure, bycopy, importcpp: "torch::nn::Dropout".} = object of ModuleHolder
    options*{.importc.}: DropoutOptions
  Dropout2d* {.pure, bycopy, importcpp: "torch::nn::Dropout2d".} = object of ModuleHolder
    options*{.importc.}: DropoutOptions
  Dropout3d* {.pure, bycopy, importcpp: "torch::nn::Dropout3d".} = object of ModuleHolder
    options*{.importc.}: DropoutOptions

  SomeDropout* = Dropout or Dropout2d or Dropout3d

func init*(T: type Dropout, proba = 0.5): T {.constructor, importcpp:"torch::nn::Dropout(@)".}
func init*(T: type Dropout2d, proba = 0.5): T {.constructor, importcpp:"torch::nn::Dropout2d(@)".}
func init*(T: type Dropout3d, proba = 0.5): T {.constructor, importcpp:"torch::nn::Dropout3d(@)".}

func forward*(dropout: SomeDropout, input: Tensor): Tensor {.importcpp: "#->forward(#)".}
