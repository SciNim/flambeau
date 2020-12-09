{.push header: "nn/options/activation.h".}


# Constructors and methods
proc constructor_SELUOptions*(inplace: bool): SELUOptions {.constructor,importcpp: "SELUOptions(@)".}

proc constructor_GLUOptions*(dim: cint): GLUOptions {.constructor,importcpp: "GLUOptions(@)".}

proc constructor_HardshrinkOptions*(lambda: cdouble): HardshrinkOptions {.constructor,importcpp: "HardshrinkOptions(@)".}

proc constructor_SoftmaxOptions*(dim: cint): SoftmaxOptions {.constructor,importcpp: "SoftmaxOptions(@)".}

proc constructor_SoftmaxFuncOptions*(dim: cint): SoftmaxFuncOptions {.constructor,importcpp: "SoftmaxFuncOptions(@)".}

proc constructor_SoftminOptions*(dim: cint): SoftminOptions {.constructor,importcpp: "SoftminOptions(@)".}

proc constructor_SoftminFuncOptions*(dim: cint): SoftminFuncOptions {.constructor,importcpp: "SoftminFuncOptions(@)".}

proc constructor_LogSoftmaxOptions*(dim: cint): LogSoftmaxOptions {.constructor,importcpp: "LogSoftmaxOptions(@)".}

proc constructor_LogSoftmaxFuncOptions*(dim: cint): LogSoftmaxFuncOptions {.constructor,importcpp: "LogSoftmaxFuncOptions(@)".}

proc constructor_ReLUOptions*(inplace: bool): ReLUOptions {.constructor,importcpp: "ReLUOptions(@)".}

proc constructor_ReLU6Options*(inplace: bool): ReLU6Options {.constructor,importcpp: "ReLU6Options(@)".}

proc constructor_SoftshrinkOptions*(lambda: cdouble): SoftshrinkOptions {.constructor,importcpp: "SoftshrinkOptions(@)".}

proc constructor_ThresholdOptions*(threshold: cdouble, value: cdouble): ThresholdOptions {.constructor,importcpp: "ThresholdOptions(@)".}

proc constructor_MultiheadAttentionOptions*(embed_dim: cint, num_heads: cint): MultiheadAttentionOptions {.constructor,importcpp: "MultiheadAttentionOptions(@)".}

proc constructor_MultiheadAttentionForwardFuncOptions*(embed_dim_to_check: cint, num_heads: cint, in_proj_weight: cint, in_proj_bias: cint, bias_k: cint, bias_v: cint, add_zero_attn: bool, dropout_p: cdouble, out_proj_weight: cint, out_proj_bias: cint): MultiheadAttentionForwardFuncOptions {.constructor,importcpp: "MultiheadAttentionForwardFuncOptions(@)".}

proc TORCH_ARG*(this: var ELUOptions): int  {.importcpp: "TORCH_ARG".}
  ## The `alpha` value for the ELU formulation. Default: 1.0

proc TORCH_ARG*(this: var ELUOptions): int  {.importcpp: "TORCH_ARG".}
  ## can optionally do the operation in-place. Default: False

proc TORCH_ARG*(this: var SELUOptions): int  {.importcpp: "TORCH_ARG".}
  ## can optionally do the operation in-place. Default: False

proc TORCH_ARG*(this: var GLUOptions): int  {.importcpp: "TORCH_ARG".}
  ## the dimension on which to split the input. Default: -1

proc TORCH_ARG*(this: var HardshrinkOptions): int  {.importcpp: "TORCH_ARG".}
  ## the `lambda` value for the Hardshrink formulation. Default: 0.5

proc TORCH_ARG*(this: var HardtanhOptions): int  {.importcpp: "TORCH_ARG".}
  ## minimum value of the linear region range. Default: -1

proc TORCH_ARG*(this: var HardtanhOptions): int  {.importcpp: "TORCH_ARG".}
  ## maximum value of the linear region range. Default: 1

proc TORCH_ARG*(this: var HardtanhOptions): int  {.importcpp: "TORCH_ARG".}
  ## can optionally do the operation in-place. Default: False

proc TORCH_ARG*(this: var LeakyReLUOptions): int  {.importcpp: "TORCH_ARG".}
  ## Controls the angle of the negative slope. Default: 1e-2

proc TORCH_ARG*(this: var LeakyReLUOptions): int  {.importcpp: "TORCH_ARG".}
  ## can optionally do the operation in-place. Default: False

proc TORCH_ARG*(this: var SoftmaxOptions): int  {.importcpp: "TORCH_ARG".}
  ## Dimension along which Softmax will be computed.

proc TORCH_ARG*(this: var SoftmaxFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## Dimension along which Softmax will be computed.

proc TORCH_ARG*(this: var SoftminOptions): int  {.importcpp: "TORCH_ARG".}
  ## Dimension along which Softmin will be computed.

proc TORCH_ARG*(this: var SoftminFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## Dimension along which Softmin will be computed.

proc TORCH_ARG*(this: var LogSoftmaxOptions): int  {.importcpp: "TORCH_ARG".}
  ## Dimension along which LogSoftmax will be computed.

proc TORCH_ARG*(this: var LogSoftmaxFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## Dimension along which LogSoftmax will be computed.

proc TORCH_ARG*(this: var PReLUOptions): int  {.importcpp: "TORCH_ARG".}
  ## number of `a` to learn. Although it takes an int as input, there is
  ## only two values are legitimate: 1, or the number of channels at input.
  ## Default: 1

proc TORCH_ARG*(this: var PReLUOptions): int  {.importcpp: "TORCH_ARG".}
  ## the initial value of `a`. Default: 0.25

proc TORCH_ARG*(this: var ReLUOptions): int  {.importcpp: "TORCH_ARG".}
  ## can optionally do the operation in-place. Default: False

proc TORCH_ARG*(this: var ReLU6Options): int  {.importcpp: "TORCH_ARG".}
  ## can optionally do the operation in-place. Default: False

proc TORCH_ARG*(this: var RReLUOptions): int  {.importcpp: "TORCH_ARG".}
  ## lower bound of the uniform distribution. Default: 1/8

proc TORCH_ARG*(this: var RReLUOptions): int  {.importcpp: "TORCH_ARG".}
  ## upper bound of the uniform distribution. Default: 1/3

proc TORCH_ARG*(this: var RReLUOptions): int  {.importcpp: "TORCH_ARG".}
  ## can optionally do the operation in-place. Default: False

proc TORCH_ARG*(this: var RReLUFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## lower bound of the uniform distribution. Default: 1/8

proc TORCH_ARG*(this: var RReLUFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## upper bound of the uniform distribution. Default: 1/3

proc TORCH_ARG*(this: var RReLUFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var RReLUFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## can optionally do the operation in-place. Default: False

proc TORCH_ARG*(this: var CELUOptions): int  {.importcpp: "TORCH_ARG".}
  ## The `alpha` value for the CELU formulation. Default: 1.0

proc TORCH_ARG*(this: var CELUOptions): int  {.importcpp: "TORCH_ARG".}
  ## can optionally do the operation in-place. Default: False

proc TORCH_ARG*(this: var SoftplusOptions): int  {.importcpp: "TORCH_ARG".}
  ## the `beta` value for the Softplus formulation. Default: 1

proc TORCH_ARG*(this: var SoftplusOptions): int  {.importcpp: "TORCH_ARG".}
  ## values above this revert to a linear function. Default: 20

proc TORCH_ARG*(this: var SoftshrinkOptions): int  {.importcpp: "TORCH_ARG".}
  ## the `lambda` value for the Softshrink formulation. Default: 0.5

proc TORCH_ARG*(this: var ThresholdOptions): int  {.importcpp: "TORCH_ARG".}
  ## The value to threshold at

proc TORCH_ARG*(this: var ThresholdOptions): int  {.importcpp: "TORCH_ARG".}
  ## The value to replace with

proc TORCH_ARG*(this: var ThresholdOptions): int  {.importcpp: "TORCH_ARG".}
  ## can optionally do the operation in-place. Default: False

proc TORCH_ARG*(this: var GumbelSoftmaxFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## non-negative scalar temperature

proc TORCH_ARG*(this: var GumbelSoftmaxFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## returned samples will be discretized as one-hot vectors, but will be
  ## differentiated as if it is the soft sample in autograd. Default: False

proc TORCH_ARG*(this: var GumbelSoftmaxFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## dimension along which softmax will be computed. Default: -1

proc TORCH_ARG*(this: var MultiheadAttentionOptions): int  {.importcpp: "TORCH_ARG".}
  ## total dimension of the model.

proc TORCH_ARG*(this: var MultiheadAttentionOptions): int  {.importcpp: "TORCH_ARG".}
  ## parallel attention heads.

proc TORCH_ARG*(this: var MultiheadAttentionOptions): int  {.importcpp: "TORCH_ARG".}
  ## a Dropout layer on attn_output_weights. Default: 0.0.

proc TORCH_ARG*(this: var MultiheadAttentionOptions): int  {.importcpp: "TORCH_ARG".}
  ## add bias as module parameter. Default: true.

proc TORCH_ARG*(this: var MultiheadAttentionOptions): int  {.importcpp: "TORCH_ARG".}
  ## add bias to the key and value sequences at dim=0.

proc TORCH_ARG*(this: var MultiheadAttentionOptions): int  {.importcpp: "TORCH_ARG".}
  ## add a new batch of zeros to the key and value sequences at dim=1.

proc TORCH_ARG*(this: var MultiheadAttentionOptions): int  {.importcpp: "TORCH_ARG".}
  ## total number of features in key. Default: c10::nullopt.

proc TORCH_ARG*(this: var MultiheadAttentionOptions): int  {.importcpp: "TORCH_ARG".}
  ## total number of features in key. Default: c10::nullopt.

proc TORCH_ARG*(this: var MultiheadAttentionForwardFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var MultiheadAttentionForwardFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var MultiheadAttentionForwardFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var MultiheadAttentionForwardFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var MultiheadAttentionForwardFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var MultiheadAttentionForwardFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var MultiheadAttentionForwardFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var MultiheadAttentionForwardFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var MultiheadAttentionForwardFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var MultiheadAttentionForwardFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var MultiheadAttentionForwardFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var MultiheadAttentionForwardFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var MultiheadAttentionForwardFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var MultiheadAttentionForwardFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var MultiheadAttentionForwardFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var MultiheadAttentionForwardFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var MultiheadAttentionForwardFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var MultiheadAttentionForwardFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var MultiheadAttentionForwardFuncOptions): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var MultiheadAttentionForwardFuncOptions): int  {.importcpp: "TORCH_ARG".}

{.pop.} # header: "nn/options/activation.h
