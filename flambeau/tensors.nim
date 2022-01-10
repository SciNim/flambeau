import raw/bindings/[rawtensors]

type
  Tensor*[T]  = distinct RawTensor

func toTensor*[T](oa: openArray[T]): Tensor[T]  =
  discard
