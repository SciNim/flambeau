# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  ../raw_bindings/tensors

# #######################################################################
#
#               Interop between Torch and Nim
#
# #######################################################################

# ArrayRefs
# -----------------------------------------------------
# libtorch/include/c10/util/ArrayRef.h

template asNimView*[T](ar: ArrayRef[T]): openarray[T] =
  toOpenArray(ar.data.unsafeAddr, 0, ar.size.int - 1)

template asTorchView*[T](oa: openarray[T]): ArrayRef[T] =
  ArrayRef[T].init(oa)

# Tensor <-> Nim sequences
# -----------------------------------------------------

# func toTensor[T](oa: openarray[T]): Tensor =
