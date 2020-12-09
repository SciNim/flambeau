{.push header: "nn/utils/rnn.h".}


# Constructors and methods
proc constructor_PackedSequence*(data: cint, batch_sizes: cint, sorted_indices: cint, unsorted_indices: cint): PackedSequence {.constructor,importcpp: "PackedSequence(@)".}

proc data*(this: PackedSequence): int  {.importcpp: "data".}

proc batch_sizes*(this: PackedSequence): int  {.importcpp: "batch_sizes".}

proc sorted_indices*(this: PackedSequence): int  {.importcpp: "sorted_indices".}

proc unsorted_indices*(this: PackedSequence): int  {.importcpp: "unsorted_indices".}

proc pin_memory*(this: PackedSequence): torch::nn::utils::rnn::PackedSequence  {.importcpp: "pin_memory".}

proc to*(this: PackedSequence, options: cint): torch::nn::utils::rnn::PackedSequence  {.importcpp: "to".}

proc cuda*(this: PackedSequence): torch::nn::utils::rnn::PackedSequence  {.importcpp: "cuda".}

proc cpu*(this: PackedSequence): torch::nn::utils::rnn::PackedSequence  {.importcpp: "cpu".}

proc is_cuda*(this: PackedSequence): bool  {.importcpp: "is_cuda".}
  ## Returns true if `data_` stored on a gpu

proc is_pinned*(this: PackedSequence): bool  {.importcpp: "is_pinned".}
  ## Returns true if `data_` stored on in pinned memory

{.pop.} # header: "nn/utils/rnn.h
