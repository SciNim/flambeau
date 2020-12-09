{.push header: "nn/options/embedding.h".}


# Constructors and methods
proc constructor_EmbeddingOptions*(num_embeddings: cint, embedding_dim: cint): EmbeddingOptions {.constructor,importcpp: "EmbeddingOptions(@)".}

proc constructor_EmbeddingBagOptions*(num_embeddings: cint, embedding_dim: cint): EmbeddingBagOptions {.constructor,importcpp: "EmbeddingBagOptions(@)".}

proc TORCH_ARG*(this: var EmbeddingOptions): int  {.importcpp: "TORCH_ARG".}
  ## The size of the dictionary of embeddings.

proc TORCH_ARG*(this: var EmbeddingOptions): int  {.importcpp: "TORCH_ARG".}
  ## The size of each embedding vector.

proc TORCH_ARG*(this: var EmbeddingOptions): int  {.importcpp: "TORCH_ARG".}
  ## The p of the p-norm to compute for the `max_norm` option. Default
  ## ``2``.

proc TORCH_ARG*(this: var EmbeddingOptions): int  {.importcpp: "TORCH_ARG".}
  ## If given, this will scale gradients by the inverse of frequency of the
  ## words in the mini-batch. Default ``false``.

proc TORCH_ARG*(this: var EmbeddingOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``true``, gradient w.r.t. `weight` matrix will be a sparse tensor.

proc TORCH_ARG*(this: var EmbeddingOptions): int  {.importcpp: "TORCH_ARG".}
  ## The learnable weights of the module of shape (num_embeddings,
  ## embedding_dim)

proc TORCH_ARG*(this: var EmbeddingFromPretrainedOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``true``, the tensor does not get updated in the learning process.
  ## Equivalent to ``embedding.weight.requires_grad_(false)``. Default:
  ## ``true``

proc TORCH_ARG*(this: var EmbeddingFromPretrainedOptions): int  {.importcpp: "TORCH_ARG".}
  ## The p of the p-norm to compute for the `max_norm` option. Default
  ## ``2``.

proc TORCH_ARG*(this: var EmbeddingFromPretrainedOptions): int  {.importcpp: "TORCH_ARG".}
  ## If given, this will scale gradients by the inverse of frequency of the
  ## words in the mini-batch. Default ``false``.

proc TORCH_ARG*(this: var EmbeddingFromPretrainedOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``true``, gradient w.r.t. `weight` matrix will be a sparse tensor.

proc TORCH_ARG*(this: var EmbeddingFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## The p of the p-norm to compute for the `max_norm` option. Default
  ## ``2``.

proc TORCH_ARG*(this: var EmbeddingFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## If given, this will scale gradients by the inverse of frequency of the
  ## words in the mini-batch. Default ``false``.

proc TORCH_ARG*(this: var EmbeddingFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``true``, gradient w.r.t. `weight` matrix will be a sparse tensor.

proc TORCH_ARG*(this: var EmbeddingBagOptions): int  {.importcpp: "TORCH_ARG".}
  ## The size of the dictionary of embeddings.

proc TORCH_ARG*(this: var EmbeddingBagOptions): int  {.importcpp: "TORCH_ARG".}
  ## The size of each embedding vector.

proc TORCH_ARG*(this: var EmbeddingBagOptions): int  {.importcpp: "TORCH_ARG".}
  ## The p of the p-norm to compute for the `max_norm` option. Default
  ## ``2``.

proc TORCH_ARG*(this: var EmbeddingBagOptions): int  {.importcpp: "TORCH_ARG".}
  ## If given, this will scale gradients by the inverse of frequency of the
  ## words in the mini-batch. Default ``false``. Note: this option is not
  ## supported when ``mode="kMax"``.

proc TORCH_ARG*(this: var EmbeddingBagOptions): int  {.importcpp: "TORCH_ARG".}
  ## ``"kSum"``, ``"kMean"`` or ``"kMax"``. Specifies the way to reduce the
  ## bag. ``"kSum"`` computes the weighted sum, taking `per_sample_weights`
  ## into consideration. ``"kMean"`` computes the average of the values in
  ## the bag, ``"kMax"`` computes the max value over each bag.

proc TORCH_ARG*(this: var EmbeddingBagOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``true``, gradient w.r.t. `weight` matrix will be a sparse tensor.
  ## Note: this option is not supported when ``mode="kMax"``.

proc TORCH_ARG*(this: var EmbeddingBagOptions): int  {.importcpp: "TORCH_ARG".}
  ## The learnable weights of the module of shape (num_embeddings,
  ## embedding_dim)

proc TORCH_ARG*(this: var EmbeddingBagOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``true``, `offsets` has one additional element, where the last
  ## element is equivalent to the size of `indices`. This matches the CSR
  ## format.

proc TORCH_ARG*(this: var EmbeddingBagFromPretrainedOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``true``, the tensor does not get updated in the learning process.
  ## Equivalent to ``embeddingbag.weight.requires_grad_(false)``. Default:
  ## ``true``

proc TORCH_ARG*(this: var EmbeddingBagFromPretrainedOptions): int  {.importcpp: "TORCH_ARG".}
  ## The p of the p-norm to compute for the `max_norm` option. Default
  ## ``2``.

proc TORCH_ARG*(this: var EmbeddingBagFromPretrainedOptions): int  {.importcpp: "TORCH_ARG".}
  ## If given, this will scale gradients by the inverse of frequency of the
  ## words in the mini-batch. Default ``false``. Note: this option is not
  ## supported when ``mode="kMax"``.

proc TORCH_ARG*(this: var EmbeddingBagFromPretrainedOptions): int  {.importcpp: "TORCH_ARG".}
  ## ``"kSum"``, ``"kMean"`` or ``"kMax"``. Specifies the way to reduce the
  ## bag. ``"kSum"`` computes the weighted sum, taking `per_sample_weights`
  ## into consideration. ``"kMean"`` computes the average of the values in
  ## the bag, ``"kMax"`` computes the max value over each bag.

proc TORCH_ARG*(this: var EmbeddingBagFromPretrainedOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``true``, gradient w.r.t. `weight` matrix will be a sparse tensor.
  ## Note: this option is not supported when ``mode="kMax"``.

proc TORCH_ARG*(this: var EmbeddingBagFromPretrainedOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``true``, `offsets` has one additional element, where the last
  ## element is equivalent to the size of `indices`. This matches the CSR
  ## format. Note: this option is currently only supported when
  ## ``mode="sum"``.

proc TORCH_ARG*(this: var EmbeddingBagFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## Only used when `input` is 1D. `offsets` determines the starting index
  ## position of each bag (sequence) in `input`.

proc TORCH_ARG*(this: var EmbeddingBagFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## The p of the p-norm to compute for the `max_norm` option. Default
  ## ``2``.

proc TORCH_ARG*(this: var EmbeddingBagFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## If given, this will scale gradients by the inverse of frequency of the
  ## words in the mini-batch. Default ``false``. Note: this option is not
  ## supported when ``mode="kMax"``.

proc TORCH_ARG*(this: var EmbeddingBagFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## ``"kSum"``, ``"kMean"`` or ``"kMax"``. Specifies the way to reduce the
  ## bag. ``"kSum"`` computes the weighted sum, taking `per_sample_weights`
  ## into consideration. ``"kMean"`` computes the average of the values in
  ## the bag, ``"kMax"`` computes the max value over each bag.

proc TORCH_ARG*(this: var EmbeddingBagFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``true``, gradient w.r.t. `weight` matrix will be a sparse tensor.
  ## Note: this option is not supported when ``mode="kMax"``.

proc TORCH_ARG*(this: var EmbeddingBagFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## a tensor of float / double weights, or None to indicate all weights
  ## should be taken to be 1. If specified, `per_sample_weights` must have
  ## exactly the same shape as input and is treated as having the same
  ## `offsets`, if those are not None.

proc TORCH_ARG*(this: var EmbeddingBagFuncOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``true``, `offsets` has one additional element, where the last
  ## element is equivalent to the size of `indices`. This matches the CSR
  ## format. Note: this option is currently only supported when
  ## ``mode="sum"``.

{.pop.} # header: "nn/options/embedding.h
