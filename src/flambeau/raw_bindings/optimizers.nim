# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  ../cpp/std_cpp,
  ../libtorch,
  ./tensors

# (Almost) raw bindings to PyTorch optimizers
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
#                       LibTorch Optimizers API
#
# #######################################################################
# libtorch/include/torch/csrc/api/include/torch/optim/optimizer.h

type
  OptimizerOptions*
    {.pure, inheritable, bycopy,
    importcpp: "torch::optim::OptimizerOptions".}
    = object

  Optimizer*
    {.pure, inheritable, bycopy,
    importcpp: "torch::optim::Optimizer".}
    = object

  SGD*
    {.pure, bycopy, importcpp: "torch::optim::SGD".}
    = object of Optimizer

func step*(optim: var Optimizer){.importcpp: "#.step()".}
func zero_grad*(optim: var Optimizer){.importcpp: "#.zero_grad()".}

func init*(
       Optim: type SGD,
       params: CppVector[Tensor],
       learning_rate: float64
     ): Optim
     {.constructor, noInit, importcpp:"torch::optim::SGD(@)".}

# SGD-specific
# -----------------------------------------------------------
type
  SGDOptions*
    {.pure, bycopy, importcpp:"torch::optim::SGDOptions".}
    = object of OptimizerOptions

func init*(T: type SGDOptions, learning_rate: float64): T {.constructor, importcpp: "torch::optim::SGDOptions(@)".}
func momentum*(opt: SGDOptions, momentum: float64): SGDOptions {.importcpp: "#.momentum(#)".}
func dampening*(opt: SGDOptions, dampening: float64): SGDOptions {.importcpp: "#.dampening(#)".}
func weight_decay*(opt: SGDOptions, weight_decay: float64): SGDOptions {.importcpp: "#.weight_decay(#)".}
func nesterov*(opt: SGDOptions, useNesterov: bool): SGDOptions {.importcpp: "#.nesterov(#)".}

func init*(
       T: type SGD,
       params: CppVector[Tensor],
       options: SGDOptions
     ): T
     {.constructor, noInit, importcpp:"torch::optim::SGD(@)".}
