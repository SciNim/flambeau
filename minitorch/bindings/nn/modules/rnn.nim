{.push header: "nn/modules/rnn.h".}


# Constructors and methods
proc constructor_RNNImplBase<Derived>*(options_: cint): RNNImplBase {.constructor,importcpp: "RNNImplBase<Derived>(@)".}

proc constructor_RNNImpl*(input_size: int64_t, hidden_size: int64_t): RNNImpl {.constructor,importcpp: "RNNImpl(@)".}

proc constructor_RNNImpl*(options_: cint): RNNImpl {.constructor,importcpp: "RNNImpl(@)".}

proc constructor_LSTMImpl*(input_size: int64_t, hidden_size: int64_t): LSTMImpl {.constructor,importcpp: "LSTMImpl(@)".}

proc constructor_LSTMImpl*(options_: cint): LSTMImpl {.constructor,importcpp: "LSTMImpl(@)".}

proc constructor_GRUImpl*(input_size: int64_t, hidden_size: int64_t): GRUImpl {.constructor,importcpp: "GRUImpl(@)".}

proc constructor_GRUImpl*(options_: cint): GRUImpl {.constructor,importcpp: "GRUImpl(@)".}

proc constructor_RNNCellImplBase<Derived>*(options_: cint): RNNCellImplBase {.constructor,importcpp: "RNNCellImplBase<Derived>(@)".}

proc constructor_RNNCellImpl*(input_size: int64_t, hidden_size: int64_t): RNNCellImpl {.constructor,importcpp: "RNNCellImpl(@)".}

proc constructor_RNNCellImpl*(options_: cint): RNNCellImpl {.constructor,importcpp: "RNNCellImpl(@)".}

proc constructor_LSTMCellImpl*(input_size: int64_t, hidden_size: int64_t): LSTMCellImpl {.constructor,importcpp: "LSTMCellImpl(@)".}

proc constructor_LSTMCellImpl*(options_: cint): LSTMCellImpl {.constructor,importcpp: "LSTMCellImpl(@)".}

proc constructor_GRUCellImpl*(input_size: int64_t, hidden_size: int64_t): GRUCellImpl {.constructor,importcpp: "GRUCellImpl(@)".}

proc constructor_GRUCellImpl*(options_: cint): GRUCellImpl {.constructor,importcpp: "GRUCellImpl(@)".}

proc reset*(this: var RNNImplBase)  {.importcpp: "reset".}
  ## Initializes the parameters of the RNN module.

proc reset_parameters*(this: var RNNImplBase)  {.importcpp: "reset_parameters".}

proc to*(this: var RNNImplBase, device: cint, dtype: cint, non_blocking: bool)  {.importcpp: "to".}
  ## Overrides `nn::Module::to()` to call `flatten_parameters()` after the
  ## original operation.

proc to*(this: var RNNImplBase, dtype: cint, non_blocking: bool)  {.importcpp: "to".}

proc to*(this: var RNNImplBase, device: cint, non_blocking: bool)  {.importcpp: "to".}

proc pretty_print*(this: RNNImplBase, stream: var std::ostream)  {.importcpp: "pretty_print".}
  ## Pretty prints the RNN module into the given `stream`.

proc flatten_parameters*(this: var RNNImplBase)  {.importcpp: "flatten_parameters".}
  ## Modifies the internal storage of weights for optimization purposes.

proc all_weights*(this: RNNImplBase): int  {.importcpp: "all_weights".}

proc reset_flat_weights*(this: var RNNImplBase)  {.importcpp: "reset_flat_weights".}

proc check_input*(this: RNNImplBase, input: cint, batch_sizes: cint)  {.importcpp: "check_input".}

proc get_expected_hidden_size*(this: RNNImplBase, input: cint, batch_sizes: cint): std::tuple<int64_t, int64_t, int64_t>  {.importcpp: "get_expected_hidden_size".}

proc check_hidden_size*(this: RNNImplBase, hx: cint, expected_hidden_size: std::tuple<int64_t, int64_t, int64_t>, msg: std::string)  {.importcpp: "check_hidden_size".}

proc check_forward_args*(this: RNNImplBase, input: cint, hidden: cint, batch_sizes: cint)  {.importcpp: "check_forward_args".}

proc permute_hidden*(this: RNNImplBase): int  {.importcpp: "permute_hidden".}

proc forward*(this: var RNNImpl): int  {.importcpp: "forward".}

proc FORWARD_HAS_DEFAULT_ARGS*(this: var RNNImpl): int  {.importcpp: "FORWARD_HAS_DEFAULT_ARGS".}

proc forward_helper*(this: var RNNImpl): int  {.importcpp: "forward_helper".}

proc FORWARD_HAS_DEFAULT_ARGS*(this: var LSTMImpl): int  {.importcpp: "FORWARD_HAS_DEFAULT_ARGS".}

proc check_forward_args*(this: LSTMImpl, input: cint, hidden: cint, batch_sizes: cint)  {.importcpp: "check_forward_args".}

proc permute_hidden*(this: LSTMImpl): int  {.importcpp: "permute_hidden".}

proc forward*(this: var GRUImpl): int  {.importcpp: "forward".}

proc FORWARD_HAS_DEFAULT_ARGS*(this: var GRUImpl): int  {.importcpp: "FORWARD_HAS_DEFAULT_ARGS".}

proc forward_helper*(this: var GRUImpl): int  {.importcpp: "forward_helper".}

proc reset*(this: var RNNCellImplBase)  {.importcpp: "reset".}
  ## Initializes the parameters of the RNNCell module.

proc reset_parameters*(this: var RNNCellImplBase)  {.importcpp: "reset_parameters".}

proc pretty_print*(this: RNNCellImplBase, stream: var std::ostream)  {.importcpp: "pretty_print".}
  ## Pretty prints the RNN module into the given `stream`.

proc check_forward_input*(this: RNNCellImplBase, input: cint)  {.importcpp: "check_forward_input".}

proc check_forward_hidden*(this: RNNCellImplBase, input: cint, hx: cint, hidden_label: std::string)  {.importcpp: "check_forward_hidden".}

proc get_nonlinearity_str*(this: RNNCellImplBase): std::string  {.importcpp: "get_nonlinearity_str".}

proc forward*(this: var RNNCellImpl): int  {.importcpp: "forward".}

proc FORWARD_HAS_DEFAULT_ARGS*(this: var RNNCellImpl): int  {.importcpp: "FORWARD_HAS_DEFAULT_ARGS".}

proc get_nonlinearity_str*(this: RNNCellImpl): std::string  {.importcpp: "get_nonlinearity_str".}

proc forward*(this: var LSTMCellImpl): int  {.importcpp: "forward".}

proc FORWARD_HAS_DEFAULT_ARGS*(this: var LSTMCellImpl): int  {.importcpp: "FORWARD_HAS_DEFAULT_ARGS".}

proc forward*(this: var GRUCellImpl): int  {.importcpp: "forward".}

proc FORWARD_HAS_DEFAULT_ARGS*(this: var GRUCellImpl): int  {.importcpp: "FORWARD_HAS_DEFAULT_ARGS".}

{.pop.} # header: "nn/modules/rnn.h
