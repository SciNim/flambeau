# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  ../cpp/std_cpp,
  ../../libtorch,
  ./attensors,
  ./neural_nets

# (Almost) raw bindings to PyTorch serialization
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
#                       LibTorch Serialization API
#
# #######################################################################
# libtorch/include/torch/csrc/api/include/torch/optim/optimizer.h

proc save*[T](module: CppSharedPtr[T], path: cstring){.sideeffect, importcpp:"torch::save(@)".}
