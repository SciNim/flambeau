{.push header: "nn/modules/embedding.h".}


# Constructors and methods
proc from_pretrained*(this: var Embedding, embeddings: cint, options: cint): torch::nn::Embedding  {.importcpp: "from_pretrained".}
  ## See the documentation for `torch::nn::EmbeddingFromPretrainedOptions`
  ## class to learn what optional arguments are supported for this
  ## function.

proc from_pretrained*(this: var EmbeddingBag, embeddings: cint, options: cint): torch::nn::EmbeddingBag  {.importcpp: "from_pretrained".}
  ## See the documentation for
  ## `torch::nn::EmbeddingBagFromPretrainedOptions` class to learn what
  ## optional arguments are supported for this function.

{.pop.} # header: "nn/modules/embedding.h
