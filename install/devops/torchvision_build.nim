# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#                   Build system for Torchvision
#
# ############################################################

# This file replaces CMake to build the torchvision DLL
# We use a separate DLL because building the library is really slow (a minute or so)
# Also CMake is much slower than Nim even for compiling + linking
# and a mess to deal with.

import
  std/[strutils, os, macros],
  flambeau/config,
  flambeau/libtorch

const rel_path = "../vendor/vision/torchvision/csrc/"
const cppSrcPath = currentSourcePath.rsplit(DirSep, 1)[0] & rel_path
static: echo cppSrcPath

# ############################################################
#
#                      External links
#
# ############################################################

{.passL: "-lpng -ljpeg".}

# ############################################################
#
#                       Compilation
#
# ############################################################

# TODO: for CUDA we need to fix Nim using gnu++14
when UseCuda:
  {.passC:"-D__CUDA_NO_HALF_OPERATORS__".}
  {.passC:"-DWITH_CUDA".}
  {.passC:"--expt-relaxed-constexpr".}

# PyTorch devel 1.8 + torchvision devel
# macro compileCppFiles(): untyped =
#   result = newStmtList()
#   for file in walkDirRec(cppSrcPath, relative = true):
#     if file.endsWith"*.h":
#       continue
#     if file == "vision.cpp": # requires Python.h, only provides cuda_version
#       continue
#     if file.endsWith"_test.cpp": # requires gtest
#       continue
#     if (file.startsWith("ops" & DirSep & "autocast") or
#         file.startsWith("ops" & DirSep & "cuda")) and
#         not UseCuda:
#       continue
#     let outfile = file.multireplace(($DirSep, "_"), (".cpp", ".o"))
#     # Need to use relative paths - https://github.com/nim-lang/Nim/issues/9370
#     result.add quote do:
#       {.compile: (rel_path & `file`, `outfile`).}

# PyTorch 1.7
macro compileCppFiles(): untyped =
  result = newStmtList()
  for file in walkDirRec(cppSrcPath, relative = true):
    if file.endsWith".h":
      continue
    if not file.startsWith("cpu" & DirSep) and
       not file.startsWith("cuda" & DirSep) and
       not file.startsWith("models" & DirSep):
      continue
    if file.endsWith"_test.cpp": # requires gtest
      continue
    if file.startsWith("cpu" & DirSep) and (
        not file.startsWith("cpu" & DirSep & "decoder") or
        not file.startsWith("cpu" & DirSep & "image") or
        not file.startsWith("cpu" & DirSep & "video")):
        # Skipping vision_cpu.h which requires Python.h
        # which requires Python dev headers
      continue
    if file == ("cpu" & DirSep & "video_reader" & DirSep & "VideoReader.cpp"):
      # Skipping Python.h
      continue
    if file == ("cpu" & DirSep & "image" & DirSep & "image.cpp"):
      # Skipping Python.h
      continue
    if file.startsWith("cuda" & DirSep) and
        not UseCuda:
      continue
    let outfile = file.multireplace(($DirSep, "_"), (".cpp", ".o"))
    # Need to use relative paths - https://github.com/nim-lang/Nim/issues/9370
    result.add quote do:
      {.compile: (rel_path & `file`, `outfile`).}

compileCppFiles()
