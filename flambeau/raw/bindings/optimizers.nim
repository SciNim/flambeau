# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  ../cpp/std_cpp,
  ../../libtorch,
  ./rawtensors

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
    {.pure, inheritable, bycopy, cppNonPod, importcpp: "torch::optim::OptimizerOptions".}
    = object

  Optimizer*
    {.pure, inheritable, bycopy, cppNonPod, importcpp: "torch::optim::Optimizer".}
    = object

  SGD*
    {.pure, bycopy, cppNonPod, importcpp: "torch::optim::SGD".}
    = object of Optimizer

  Adam*
    {.pure, bycopy, cppNonPod, importcpp: "torch::optim::Adam".}
    = object of Optimizer

  AdamW*
    {.pure, bycopy, cppNonPod, requiresInit,  importcpp: "torch::optim::AdamW".}
    = object of Optimizer


func step*(optim: var Optimizer){.importcpp: "#.step()".}
func zero_grad*(optim: var Optimizer){.importcpp: "#.zero_grad()".}

func initSGD*(): SGD =
  SGD()
func initAdam*(): Adam =
  Adam()
func initAdamw*(): Adamw =
  Adamw()

func init*(
       Optim: type SGD,
       params: CppVector[RawTensor],
       learning_rate: float64
     ): Optim
     {.constructor, importcpp: "torch::optim::SGD(@)".}

func init*(
       Optim: type Adam,
       params: CppVector[RawTensor],
       learning_rate: float64
     ): Optim
     {.constructor, importcpp:"torch::optim::Adam(@)".}

func init*(
       Optim: type AdamW,
       params: CppVector[RawTensor],
       learning_rate: float64
     ): Optim
     {.constructor, importcpp:"torch::optim::AdamW(@)".}


# SGD-specific
# -----------------------------------------------------------
type
  SGDOptions*
    {.pure, bycopy, importcpp: "torch::optim::SGDOptions".}
    = object of OptimizerOptions

func init*(T: type SGDOptions, learning_rate: float64): T {.constructor, importcpp: "torch::optim::SGDOptions(@)".}
func momentum*(opt: SGDOptions, momentum: float64): SGDOptions {.importcpp: "#.momentum(#)".}
func dampening*(opt: SGDOptions, dampening: float64): SGDOptions {.importcpp: "#.dampening(#)".}
func weight_decay*(opt: SGDOptions, weight_decay: float64): SGDOptions {.importcpp: "#.weight_decay(#)".}
func nesterov*(opt: SGDOptions, useNesterov: bool): SGDOptions {.importcpp: "#.nesterov(#)".}

func init*(
       T: type SGD,
       params: CppVector[RawTensor],
       options: SGDOptions
     ): T
     {.constructor, importcpp: "torch::optim::SGD(@)".}

# Adam-specific
# -----------------------------------------------------------
type
  AdamOptions*
    {.pure, bycopy, importcpp: "torch::optim::AdamOptions".}
    = object of OptimizerOptions

func init*(T: type AdamOptions, learning_rate: float64): T {.constructor, importcpp: "torch::optim::AdamOptions(@)".}
func betas*(opt: AdamOptions, beta: float64): AdamOptions {.importcpp: "#.betas(#)".}
func eps*(opt: AdamOptions, eps: float64): AdamOptions {.importcpp: "#.eps(#)".}
func weight_decay*(opt: AdamOptions, weight_decay: float64): AdamOptions {.importcpp: "#.weight_decay(#)".}
func amsgrad*(opt: AdamOptions, useAmsGrad: bool): AdamOptions {.importcpp: "#.amsgrad(#)".}

func init*(
       T: type Adam,
       params: CppVector[RawTensor],
       options: AdamOptions
     ): T
     {.constructor, noInit, importcpp: "torch::optim::Adam(@)".}

# AdamW-specific
# -----------------------------------------------------------
type
  AdamWOptions*
    {.pure, bycopy, importcpp: "torch::optim::AdamWOptions".}
    = object of OptimizerOptions

func init*(T: type AdamWOptions, learning_rate: float64): T {.constructor, importcpp: "torch::optim::AdamWOptions(@)".}
func betas*(opt: AdamWOptions, beta: float64): AdamWOptions {.importcpp: "#.betas(#)".}
func eps*(opt: AdamWOptions, eps: float64): AdamWOptions {.importcpp: "#.eps(#)".}
func weight_decay*(opt: AdamWOptions, weight_decay: float64): AdamWOptions {.importcpp: "#.weight_decay(#)".}
func amsgrad*(opt: AdamWOptions, useAmsGrad: bool): AdamWOptions {.importcpp: "#.amsgrad(#)".}

func init*(
       T: type AdamW,
       params: CppVector[RawTensor],
       options: AdamWOptions
     ): T
     {.constructor, noInit, importcpp: "torch::optim::AdamW(@)".}
