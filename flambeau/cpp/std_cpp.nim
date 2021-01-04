# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#                   C++ standard types wrapper
#
# ############################################################

# std::string
# -----------------------------------------------------------------------

{.push header: "<string>".}

type
  CppString* {.importcpp: "std::string", bycopy.} = object

func len*(s: CppString): int {.importcpp: "#.length()".}
  ## Returns the length of a C++ std::string
func data*(s: CppString): lent char {.importcpp: "#.data()".}
  ## Returns a pointer to the raw data of a C++ std::string
func cstring*(s: CppString): cstring {.importcpp: "#.c_str()"}

# Interop
# ------------------------------

func `$`*(s: CppString): string =
  result = newString(s.len)
  copyMem(result[0].addr, s.data.unsafeAddr, s.len)

{.pop.}
