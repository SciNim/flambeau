{.push header: "nn/options/padding.h".}


# Constructors and methods
proc constructor_PadFuncOptions*(pad: cint): PadFuncOptions {.constructor,importcpp: "PadFuncOptions(@)".}

proc value_*(this: var ConstantPadOptions, cint): ConstantPadOptions<D>  {.importcpp: "value_".}

proc TORCH_ARG*(this: var ConstantPadOptions): int  {.importcpp: "TORCH_ARG".}
  ## Fill value for constant padding.

proc TORCH_ARG*(this: var PadFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## m-elements tuple, where m/2 <= input dimensions and m is even.

proc TORCH_ARG*(this: var PadFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## "constant", "reflect", "replicate" or "circular". Default: "constant"

proc TORCH_ARG*(this: var PadFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## fill value for "constant" padding. Default: 0

{.pop.} # header: "nn/options/padding.h
