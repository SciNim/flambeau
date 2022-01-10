import std/[os, macros]
const dummyHeader = "-I" & currentSourcePath().parentDir()
static:
  echo dummyHeader

{.passC: dummyHeader}

type
  RawTensor* {.header:"dummyTorch.h", importcpp: "torch::Tensor", cppNonPod, bycopy.} = object
  Tensor*[T]  = distinct RawTensor

func toTensor*[T](oa: openArray[T]): Tensor[T]  =
  discard
