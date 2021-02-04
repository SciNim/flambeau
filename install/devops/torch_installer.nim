# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/[asyncdispatch, httpclient,
       strformat, strutils, os, times],
  zip/zipfiles

type
  Acceleration* = enum
    Cpu = "cpu"
    Cuda92 = "cu92"
    Cuda101 = "cu101"
    Cuda102 = "cu102"
    Cuda110 = "cu110"

  ABI* = enum
    Cpp = "shared-with-deps"
    Cpp11 = "cxx11-abi-shared-with-deps"

proc getProjectDir(): string {.compileTime.} =
  currentSourcePath.rsplit(DirSep, 1)[0]

proc onProgressChanged(total, progress, speed: BiggestInt) {.async.} =
  echo &"Downloaded {progress} of {total}"
  echo &"Current rate: {speed.float64 / (1000*1000):4.3f} MiBi/s" # TODO the unit is neither MB or Mb or MiBi ???

proc downloadTo(url, targetDir, filename: string) {.async.} =
  var client = newAsyncHttpClient()
  defer: client.close()
  client.onProgressChanged = onProgressChanged
  echo "Starting download of \"", url, '\"'
  echo "Storing temporary into: \"", targetDir, '\"'
  await client.downloadFile(url, targetDir / filename)

proc getUrlAndFilename(version = "1.7.1", accel = Cuda110, abi = Cpp11): tuple[url, filename: string] =
  result.filename = "libtorch-"

  when defined(linux):
    result.filename &= &"{abi}-{version}"
    if accel != Cuda102:
      result.filename &= &"%2B{accel}"
    result.filename &= ".zip"
  elif defined(windows):
    doAssert abi == Cpp, "LibTorch for Windows does not support the C++11 ABI"
    result.filename &= &"-win-{abi}-{version}.zip"
  elif defined(osx):
    doAssert accel == Cpu, "LibTorch for MacOS does not support GPU acceleration"
    result.filename &= &"macos-{version}.zip"

  result.url = &"https://download.pytorch.org/libtorch/{accel}/{result.filename}"

proc genNimsConfig(includePath, libPath: string) =
  var configFile = open("install/config.nim", fmWrite)
  configFile.writeLine(&"## Generated file during torch_installer execution -- {now()}")
  configFile.writeLine("## Do not modify unless you know what you're doing")
  configFile.writeLine(&"""
switch("passC","-I{includePath}")
switch("passL","-Wl,-rpath,{libPath}")
switch("passL","-ltorch")
  """)

proc downloadLibTorch(url, targetDir, filename: string) =
  waitFor url.downloadTo(targetDir, filename)

proc uncompress(targetDir, filename: string, delete = false) =
  var z: ZipArchive
  let tmp = targetDir / filename
  if not z.open(tmp):
    raise newException(IOError, &"Could not open zip file: \"{tmp}\"")
  defer: z.close()
  echo "Decompressing \"", tmp, "\" and storing into \"", targetDir, "\""
  z.extractAll(targetDir)
  echo "Done."
  if delete:
    echo "Deleting \"", tmp, "\""
    removeFile(tmp)
  else:
    echo "Not deleting \"", tmp, "\""

  let includePath = targetDir & DirSep & "libtorch" & DirSep & "include"
  let libPath = targetDir & DirSep & "libtorch" & DirSep & "lib"
  genNimsConfig(includePath, libPath)
  # echo "[Important]: Make sure that '" & libPath & "' is in your LIBRARY_PATH."

proc localTest(targetDir: string) =
  let includePath = targetDir & DirSep & "libtorch" & DirSep & "include"
  let libPath = targetDir & DirSep & "libtorch" & DirSep & "lib"
  genNimsConfig(includePath, libPath)

when isMainModule:
  let (url, filename) = getUrlAndFilename()
  let target = getProjectDir().parentDir() & DirSep & "vendor"
  # downloadLibTorch(url, target, filename)
  # uncompress(target, filename)
  localTest(target)
