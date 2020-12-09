{.push header: "nn/modules/activation.h".}


# Constructors and methods
proc constructor_ELUImpl*(options_: cint): ELUImpl {.constructor,importcpp: "ELUImpl(@)".}

proc constructor_SELUImpl*(options_: cint): SELUImpl {.constructor,importcpp: "SELUImpl(@)".}

proc constructor_HardshrinkImpl*(options_: cint): HardshrinkImpl {.constructor,importcpp: "HardshrinkImpl(@)".}

proc constructor_HardtanhImpl*(options_: cint): HardtanhImpl {.constructor,importcpp: "HardtanhImpl(@)".}

proc constructor_LeakyReLUImpl*(options_: cint): LeakyReLUImpl {.constructor,importcpp: "LeakyReLUImpl(@)".}

proc constructor_SoftmaxImpl*(dim: cint): SoftmaxImpl {.constructor,importcpp: "SoftmaxImpl(@)".}

proc constructor_SoftmaxImpl*(options_: cint): SoftmaxImpl {.constructor,importcpp: "SoftmaxImpl(@)".}

proc constructor_SoftminImpl*(dim: cint): SoftminImpl {.constructor,importcpp: "SoftminImpl(@)".}

proc constructor_SoftminImpl*(options_: cint): SoftminImpl {.constructor,importcpp: "SoftminImpl(@)".}

proc constructor_LogSoftmaxImpl*(dim: cint): LogSoftmaxImpl {.constructor,importcpp: "LogSoftmaxImpl(@)".}

proc constructor_LogSoftmaxImpl*(options_: cint): LogSoftmaxImpl {.constructor,importcpp: "LogSoftmaxImpl(@)".}

proc constructor_PReLUImpl*(options_: cint): PReLUImpl {.constructor,importcpp: "PReLUImpl(@)".}

proc constructor_ReLUImpl*(options_: cint): ReLUImpl {.constructor,importcpp: "ReLUImpl(@)".}

proc constructor_ReLU6Impl*(options_: cint): ReLU6Impl {.constructor,importcpp: "ReLU6Impl(@)".}

proc constructor_RReLUImpl*(options_: cint): RReLUImpl {.constructor,importcpp: "RReLUImpl(@)".}

proc constructor_CELUImpl*(options_: cint): CELUImpl {.constructor,importcpp: "CELUImpl(@)".}

proc constructor_GLUImpl*(options_: cint): GLUImpl {.constructor,importcpp: "GLUImpl(@)".}

proc constructor_SoftplusImpl*(options_: cint): SoftplusImpl {.constructor,importcpp: "SoftplusImpl(@)".}

proc constructor_SoftshrinkImpl*(options_: cint): SoftshrinkImpl {.constructor,importcpp: "SoftshrinkImpl(@)".}

proc constructor_ThresholdImpl*(threshold: cdouble, value: cdouble): ThresholdImpl {.constructor,importcpp: "ThresholdImpl(@)".}

proc constructor_ThresholdImpl*(options_: cint): ThresholdImpl {.constructor,importcpp: "ThresholdImpl(@)".}

proc constructor_MultiheadAttentionImpl*(embed_dim: cint, num_heads: cint): MultiheadAttentionImpl {.constructor,importcpp: "MultiheadAttentionImpl(@)".}

proc constructor_MultiheadAttentionImpl*(options_: cint): MultiheadAttentionImpl {.constructor,importcpp: "MultiheadAttentionImpl(@)".}

proc forward*(this: var ELUImpl): int  {.importcpp: "forward".}

proc reset*(this: var ELUImpl)  {.importcpp: "reset".}

proc pretty_print*(this: ELUImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `ELU` module into the given `stream`.

proc forward*(this: var SELUImpl): int  {.importcpp: "forward".}

proc reset*(this: var SELUImpl)  {.importcpp: "reset".}

proc pretty_print*(this: SELUImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `SELU` module into the given `stream`.

proc forward*(this: var HardshrinkImpl): int  {.importcpp: "forward".}

proc reset*(this: var HardshrinkImpl)  {.importcpp: "reset".}

proc pretty_print*(this: HardshrinkImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `Hardshrink` module into the given `stream`.

proc forward*(this: var HardtanhImpl): int  {.importcpp: "forward".}

proc reset*(this: var HardtanhImpl)  {.importcpp: "reset".}

proc pretty_print*(this: HardtanhImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `Hardtanh` module into the given `stream`.

proc forward*(this: var LeakyReLUImpl): int  {.importcpp: "forward".}

proc reset*(this: var LeakyReLUImpl)  {.importcpp: "reset".}

proc pretty_print*(this: LeakyReLUImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `LeakyReLU` module into the given `stream`.

proc forward*(this: var LogSigmoidImpl): int  {.importcpp: "forward".}

proc reset*(this: var LogSigmoidImpl)  {.importcpp: "reset".}

proc pretty_print*(this: LogSigmoidImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `LogSigmoid` module into the given `stream`.

proc forward*(this: var SoftmaxImpl): int  {.importcpp: "forward".}

proc reset*(this: var SoftmaxImpl)  {.importcpp: "reset".}

proc pretty_print*(this: SoftmaxImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `Softmax` module into the given `stream`.

proc forward*(this: var SoftminImpl): int  {.importcpp: "forward".}

proc reset*(this: var SoftminImpl)  {.importcpp: "reset".}

proc pretty_print*(this: SoftminImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `Softmin` module into the given `stream`.

proc forward*(this: var LogSoftmaxImpl): int  {.importcpp: "forward".}

proc reset*(this: var LogSoftmaxImpl)  {.importcpp: "reset".}

proc pretty_print*(this: LogSoftmaxImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `LogSoftmax` module into the given `stream`.

proc forward*(this: var Softmax2dImpl): int  {.importcpp: "forward".}

proc reset*(this: var Softmax2dImpl)  {.importcpp: "reset".}

proc pretty_print*(this: Softmax2dImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `Softmax2d` module into the given `stream`.

proc forward*(this: var PReLUImpl): int  {.importcpp: "forward".}

proc reset*(this: var PReLUImpl)  {.importcpp: "reset".}

proc pretty_print*(this: PReLUImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `PReLU` module into the given `stream`.

proc forward*(this: var ReLUImpl): int  {.importcpp: "forward".}

proc reset*(this: var ReLUImpl)  {.importcpp: "reset".}

proc pretty_print*(this: ReLUImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `ReLU` module into the given `stream`.

proc forward*(this: var ReLU6Impl): int  {.importcpp: "forward".}

proc reset*(this: var ReLU6Impl)  {.importcpp: "reset".}

proc pretty_print*(this: ReLU6Impl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `ReLU6` module into the given `stream`.

proc forward*(this: var RReLUImpl): int  {.importcpp: "forward".}

proc reset*(this: var RReLUImpl)  {.importcpp: "reset".}

proc pretty_print*(this: RReLUImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `RReLU` module into the given `stream`.

proc forward*(this: var CELUImpl): int  {.importcpp: "forward".}

proc reset*(this: var CELUImpl)  {.importcpp: "reset".}

proc pretty_print*(this: CELUImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `CELU` module into the given `stream`.

proc forward*(this: var GLUImpl): int  {.importcpp: "forward".}

proc reset*(this: var GLUImpl)  {.importcpp: "reset".}

proc pretty_print*(this: GLUImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `GLU` module into the given `stream`.

proc forward*(this: var GELUImpl): int  {.importcpp: "forward".}

proc reset*(this: var GELUImpl)  {.importcpp: "reset".}

proc pretty_print*(this: GELUImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `GELU` module into the given `stream`.

proc forward*(this: var SiLUImpl): int  {.importcpp: "forward".}

proc reset*(this: var SiLUImpl)  {.importcpp: "reset".}

proc pretty_print*(this: SiLUImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `SiLU` module into the given `stream`.

proc forward*(this: var SigmoidImpl): int  {.importcpp: "forward".}

proc reset*(this: var SigmoidImpl)  {.importcpp: "reset".}

proc pretty_print*(this: SigmoidImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `Sigmoid` module into the given `stream`.

proc forward*(this: var SoftplusImpl): int  {.importcpp: "forward".}

proc reset*(this: var SoftplusImpl)  {.importcpp: "reset".}

proc pretty_print*(this: SoftplusImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `Softplus` module into the given `stream`.

proc forward*(this: var SoftshrinkImpl): int  {.importcpp: "forward".}

proc reset*(this: var SoftshrinkImpl)  {.importcpp: "reset".}

proc pretty_print*(this: SoftshrinkImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `Softshrink` module into the given `stream`.

proc forward*(this: var SoftsignImpl): int  {.importcpp: "forward".}

proc reset*(this: var SoftsignImpl)  {.importcpp: "reset".}

proc pretty_print*(this: SoftsignImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `Softsign` module into the given `stream`.

proc forward*(this: var TanhImpl): int  {.importcpp: "forward".}

proc reset*(this: var TanhImpl)  {.importcpp: "reset".}

proc pretty_print*(this: TanhImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `Tanh` module into the given `stream`.

proc forward*(this: var TanhshrinkImpl): int  {.importcpp: "forward".}

proc reset*(this: var TanhshrinkImpl)  {.importcpp: "reset".}

proc pretty_print*(this: TanhshrinkImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `Tanhshrink` module into the given `stream`.

proc forward*(this: var ThresholdImpl): int  {.importcpp: "forward".}

proc reset*(this: var ThresholdImpl)  {.importcpp: "reset".}

proc pretty_print*(this: ThresholdImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `Threshold` module into the given `stream`.

proc forward*(this: var MultiheadAttentionImpl): int  {.importcpp: "forward".}

proc FORWARD_HAS_DEFAULT_ARGS*(this: var MultiheadAttentionImpl): int  {.importcpp: "FORWARD_HAS_DEFAULT_ARGS".}

proc _reset_parameters*(this: var MultiheadAttentionImpl)  {.importcpp: "_reset_parameters".}

{.pop.} # header: "nn/modules/activation.h
