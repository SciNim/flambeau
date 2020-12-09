{.push header: "nn/options/rnn.h".}


# Constructors and methods
proc constructor_RNNOptionsBase*(mode: torch::nn::detail::RNNOptionsBase::rnn_options_base_mode_t, input_size: cint, hidden_size: cint): RNNOptionsBase {.constructor,importcpp: "RNNOptionsBase(@)".}

proc constructor_RNNOptions*(input_size: cint, hidden_size: cint): RNNOptions {.constructor,importcpp: "RNNOptions(@)".}

proc constructor_LSTMOptions*(input_size: cint, hidden_size: cint): LSTMOptions {.constructor,importcpp: "LSTMOptions(@)".}

proc constructor_GRUOptions*(input_size: cint, hidden_size: cint): GRUOptions {.constructor,importcpp: "GRUOptions(@)".}

proc constructor_RNNCellOptionsBase*(input_size: cint, hidden_size: cint, bias: bool, num_chunks: cint): RNNCellOptionsBase {.constructor,importcpp: "RNNCellOptionsBase(@)".}

proc constructor_RNNCellOptions*(input_size: cint, hidden_size: cint): RNNCellOptions {.constructor,importcpp: "RNNCellOptions(@)".}

proc constructor_LSTMCellOptions*(input_size: cint, hidden_size: cint): LSTMCellOptions {.constructor,importcpp: "LSTMCellOptions(@)".}

proc constructor_GRUCellOptions*(input_size: cint, hidden_size: cint): GRUCellOptions {.constructor,importcpp: "GRUCellOptions(@)".}

proc TORCH_ARG*(this: var RNNOptionsBase): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var RNNOptionsBase): int  {.importcpp: "TORCH_ARG".}
  ## The number of features of a single sample in the input sequence `x`.

proc TORCH_ARG*(this: var RNNOptionsBase): int  {.importcpp: "TORCH_ARG".}
  ## The number of features in the hidden state `h`.

proc TORCH_ARG*(this: var RNNOptionsBase): int  {.importcpp: "TORCH_ARG".}
  ## The number of recurrent layers (cells) to use.

proc TORCH_ARG*(this: var RNNOptionsBase): int  {.importcpp: "TORCH_ARG".}
  ## Whether a bias term should be added to all linear operations.

proc TORCH_ARG*(this: var RNNOptionsBase): int  {.importcpp: "TORCH_ARG".}
  ## If true, the input sequence should be provided as `(batch, sequence,
  ## features)`. If false (default), the expected layout is `(sequence,
  ## batch, features)`.

proc TORCH_ARG*(this: var RNNOptionsBase): int  {.importcpp: "TORCH_ARG".}
  ## If non-zero, adds dropout with the given probability to the output of
  ## each RNN layer, except the final layer.

proc TORCH_ARG*(this: var RNNOptionsBase): int  {.importcpp: "TORCH_ARG".}
  ## Whether to make the RNN bidirectional.

proc TORCH_ARG*(this: var RNNOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of expected features in the input `x`

proc TORCH_ARG*(this: var RNNOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of features in the hidden state `h`

proc TORCH_ARG*(this: var RNNOptions): int  {.importcpp: "TORCH_ARG".}
  ## Number of recurrent layers. E.g., setting ``num_layers=2`` would mean
  ## stacking two RNNs together to form a `stacked RNN`, with the second
  ## RNN taking in outputs of the first RNN and computing the final
  ## results. Default: 1

proc TORCH_ARG*(this: var RNNOptions): int  {.importcpp: "TORCH_ARG".}
  ## The non-linearity to use. Can be either ``torch::kTanh`` or
  ## ``torch::kReLU``. Default: ``torch::kTanh``

proc TORCH_ARG*(this: var RNNOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``false``, then the layer does not use bias weights `b_ih` and
  ## `b_hh`. Default: ``true``

proc TORCH_ARG*(this: var RNNOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``true``, then the input and output tensors are provided as
  ## `(batch, seq, feature)`. Default: ``false``

proc TORCH_ARG*(this: var RNNOptions): int  {.importcpp: "TORCH_ARG".}
  ## If non-zero, introduces a `Dropout` layer on the outputs of each RNN
  ## layer except the last layer, with dropout probability equal to
  ## `dropout`. Default: 0

proc TORCH_ARG*(this: var RNNOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``true``, becomes a bidirectional RNN. Default: ``false``

proc TORCH_ARG*(this: var LSTMOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of expected features in the input `x`

proc TORCH_ARG*(this: var LSTMOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of features in the hidden state `h`

proc TORCH_ARG*(this: var LSTMOptions): int  {.importcpp: "TORCH_ARG".}
  ## Number of recurrent layers. E.g., setting ``num_layers=2`` would mean
  ## stacking two LSTMs together to form a `stacked LSTM`, with the second
  ## LSTM taking in outputs of the first LSTM and computing the final
  ## results. Default: 1

proc TORCH_ARG*(this: var LSTMOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``false``, then the layer does not use bias weights `b_ih` and
  ## `b_hh`. Default: ``true``

proc TORCH_ARG*(this: var LSTMOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``true``, then the input and output tensors are provided as (batch,
  ## seq, feature). Default: ``false``

proc TORCH_ARG*(this: var LSTMOptions): int  {.importcpp: "TORCH_ARG".}
  ## If non-zero, introduces a `Dropout` layer on the outputs of each LSTM
  ## layer except the last layer, with dropout probability equal to
  ## `dropout`. Default: 0

proc TORCH_ARG*(this: var LSTMOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``true``, becomes a bidirectional LSTM. Default: ``false``

proc TORCH_ARG*(this: var GRUOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of expected features in the input `x`

proc TORCH_ARG*(this: var GRUOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of features in the hidden state `h`

proc TORCH_ARG*(this: var GRUOptions): int  {.importcpp: "TORCH_ARG".}
  ## Number of recurrent layers. E.g., setting ``num_layers=2`` would mean
  ## stacking two GRUs together to form a `stacked GRU`, with the second
  ## GRU taking in outputs of the first GRU and computing the final
  ## results. Default: 1

proc TORCH_ARG*(this: var GRUOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``false``, then the layer does not use bias weights `b_ih` and
  ## `b_hh`. Default: ``true``

proc TORCH_ARG*(this: var GRUOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``true``, then the input and output tensors are provided as (batch,
  ## seq, feature). Default: ``false``

proc TORCH_ARG*(this: var GRUOptions): int  {.importcpp: "TORCH_ARG".}
  ## If non-zero, introduces a `Dropout` layer on the outputs of each GRU
  ## layer except the last layer, with dropout probability equal to
  ## `dropout`. Default: 0

proc TORCH_ARG*(this: var GRUOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``true``, becomes a bidirectional GRU. Default: ``false``

proc TORCH_ARG*(this: var RNNCellOptionsBase): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var RNNCellOptionsBase): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var RNNCellOptionsBase): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var RNNCellOptionsBase): int  {.importcpp: "TORCH_ARG".}

proc TORCH_ARG*(this: var RNNCellOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of expected features in the input `x`

proc TORCH_ARG*(this: var RNNCellOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of features in the hidden state `h`

proc TORCH_ARG*(this: var RNNCellOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``false``, then the layer does not use bias weights `b_ih` and
  ## `b_hh`. Default: ``true``

proc TORCH_ARG*(this: var RNNCellOptions): int  {.importcpp: "TORCH_ARG".}
  ## The non-linearity to use. Can be either ``torch::kTanh`` or
  ## ``torch::kReLU``. Default: ``torch::kTanh``

proc TORCH_ARG*(this: var LSTMCellOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of expected features in the input `x`

proc TORCH_ARG*(this: var LSTMCellOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of features in the hidden state `h`

proc TORCH_ARG*(this: var LSTMCellOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``false``, then the layer does not use bias weights `b_ih` and
  ## `b_hh`. Default: ``true``

proc TORCH_ARG*(this: var GRUCellOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of expected features in the input `x`

proc TORCH_ARG*(this: var GRUCellOptions): int  {.importcpp: "TORCH_ARG".}
  ## The number of features in the hidden state `h`

proc TORCH_ARG*(this: var GRUCellOptions): int  {.importcpp: "TORCH_ARG".}
  ## If ``false``, then the layer does not use bias weights `b_ih` and
  ## `b_hh`. Default: ``true``

{.pop.} # header: "nn/options/rnn.h
