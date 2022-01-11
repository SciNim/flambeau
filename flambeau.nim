# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Raw exports
# ----------------------------------------------------------------

when not defined(cpp):
  {.error: "Flambeau requires C++ backend required to use Torch".}

import ./flambeau/tensors
#export tensors

# C++ Standard Library
# ----------------------------------------------------------------
import ./flambeau/raw/cpp/std_cpp
#export std_cpp

import ./flambeau/raw/bindings/c10
#export c10

import ./flambeau/raw/sugar/rawinterop

# I don't know why but if you remove this it doesn't work
let local_make_sure_name_is_not_used_t {.used.} = toRawTensor(@[0])
