# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

packageName   = "flambeau"
version       = "0.0.1"
author        = "Mamy André-Ratsimbazafy"
description   = "A state-of-the-art tensor and deep learning backend on CPU, Nvidia Cuda, AMD HIP, OpenCL, Vulkan, OpenGL"
license       = "MIT or Apache License 2.0"

### Dependencies
requires "nim >= 1.4.2"
requires "zip"

when defined(nimdistros):
  import distros
  foreignDep "libjpeg"
  foreignDep "libpng"
  foreignDep "ffmpeg"
  foreignDep "gtest" # TODO remove this dependency (due to io/decoder/sync_decoder_test.cpp)

### Build
task build_torchvision, "Build the dependency torchvision":
  when defined(windows):
    const libName = "torchvision.dll"
  elif defined(macosx):
    const libName = "libtorchvision.dylib"
  else:
    const libName = "libtorchvision.so"

  const libBuilder = "install/torchvision_build.nim"

  if not dirExists "vendor/lib":
    mkDir "vendor/lib"
  switch("out", "vendor/lib/" & libName)
  switch("define", "danger")
  switch("app", "lib")
  switch("noMain")
  switch("gc", "none")
  setCommand "cpp", libBuilder

task install_libtorch, "Download and install libtorch":
  switch("skipParentCfg", "on")
  const libInstaller = "install/torch_installer.nim"
  setCommand "cpp", libInstaller
  switch("run")

before install:
  install_libtorchTask()

before develop:
  install_libtorchTask()
