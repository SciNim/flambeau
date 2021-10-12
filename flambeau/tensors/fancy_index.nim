# Copyright 2017-2021 the Flambeau contributors

import ../tensors
# Reuse same desugar syntax as RawTensor
include ../raw/sugar/indexing

macro `[]`*[T](t: Tensor[T], args: varargs[untyped]): untyped =
  let new_args = getAST(desugarSlices(args))
  result = quote do:
    slice_typed_dispatch(`t`, `new_args`)

macro `[]=`*[T](t: var Tensor[T], args: varargs[untyped]): untyped =
  var tmp = args
  let val = tmp.pop
  let new_args = getAST(desugarSlices(tmp))

  result = quote do:
    slice_typed_dispatch_mut(`t`, `new_args`, `val`)
