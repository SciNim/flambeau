import std/[os, macros]
const dummyHeader = "-I" & currentSourcePath().parentDir()
static:
  echo dummyHeader

{.passC: dummyHeader}

type
  RawTensor* {.header:"dummyTorch.h", importcpp: "torch::Tensor".} = object
  Tensor*[T]  = distinct RawTensor

func initRawTensor*() : RawTensor {.constructor, importcpp: "torch::Tensor".}
func toTensor*[T](oa: openArray[T]): Tensor[T]  =
  discard
