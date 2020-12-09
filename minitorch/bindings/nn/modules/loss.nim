{.push header: "nn/modules/loss.h".}


# Constructors and methods
proc constructor_L1LossImpl*(options_: cint): L1LossImpl {.constructor,importcpp: "L1LossImpl(@)".}

proc constructor_KLDivLossImpl*(options_: cint): KLDivLossImpl {.constructor,importcpp: "KLDivLossImpl(@)".}

proc constructor_MSELossImpl*(options_: cint): MSELossImpl {.constructor,importcpp: "MSELossImpl(@)".}

proc constructor_BCELossImpl*(options_: cint): BCELossImpl {.constructor,importcpp: "BCELossImpl(@)".}

proc constructor_HingeEmbeddingLossImpl*(options_: cint): HingeEmbeddingLossImpl {.constructor,importcpp: "HingeEmbeddingLossImpl(@)".}

proc constructor_MultiMarginLossImpl*(options_: cint): MultiMarginLossImpl {.constructor,importcpp: "MultiMarginLossImpl(@)".}

proc constructor_CosineEmbeddingLossImpl*(options_: cint): CosineEmbeddingLossImpl {.constructor,importcpp: "CosineEmbeddingLossImpl(@)".}

proc constructor_SmoothL1LossImpl*(options_: cint): SmoothL1LossImpl {.constructor,importcpp: "SmoothL1LossImpl(@)".}

proc constructor_MultiLabelMarginLossImpl*(options_: cint): MultiLabelMarginLossImpl {.constructor,importcpp: "MultiLabelMarginLossImpl(@)".}

proc constructor_SoftMarginLossImpl*(options_: cint): SoftMarginLossImpl {.constructor,importcpp: "SoftMarginLossImpl(@)".}

proc constructor_MultiLabelSoftMarginLossImpl*(options_: cint): MultiLabelSoftMarginLossImpl {.constructor,importcpp: "MultiLabelSoftMarginLossImpl(@)".}

proc constructor_TripletMarginLossImpl*(options_: cint): TripletMarginLossImpl {.constructor,importcpp: "TripletMarginLossImpl(@)".}

proc constructor_CTCLossImpl*(options_: cint): CTCLossImpl {.constructor,importcpp: "CTCLossImpl(@)".}

proc constructor_PoissonNLLLossImpl*(options_: cint): PoissonNLLLossImpl {.constructor,importcpp: "PoissonNLLLossImpl(@)".}

proc constructor_MarginRankingLossImpl*(options_: cint): MarginRankingLossImpl {.constructor,importcpp: "MarginRankingLossImpl(@)".}

proc constructor_NLLLossImpl*(options_: cint): NLLLossImpl {.constructor,importcpp: "NLLLossImpl(@)".}

proc constructor_CrossEntropyLossImpl*(options_: cint): CrossEntropyLossImpl {.constructor,importcpp: "CrossEntropyLossImpl(@)".}

proc constructor_BCEWithLogitsLossImpl*(options_: cint): BCEWithLogitsLossImpl {.constructor,importcpp: "BCEWithLogitsLossImpl(@)".}

proc reset*(this: var L1LossImpl)  {.importcpp: "reset".}

proc pretty_print*(this: L1LossImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `L1Loss` module into the given `stream`.

proc forward*(this: var L1LossImpl): int  {.importcpp: "forward".}

proc reset*(this: var KLDivLossImpl)  {.importcpp: "reset".}

proc pretty_print*(this: KLDivLossImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `KLDivLoss` module into the given `stream`.

proc forward*(this: var KLDivLossImpl): int  {.importcpp: "forward".}

proc reset*(this: var MSELossImpl)  {.importcpp: "reset".}

proc pretty_print*(this: MSELossImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `MSELoss` module into the given `stream`.

proc forward*(this: var MSELossImpl): int  {.importcpp: "forward".}

proc reset*(this: var BCELossImpl)  {.importcpp: "reset".}

proc pretty_print*(this: BCELossImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `BCELoss` module into the given `stream`.

proc forward*(this: var BCELossImpl): int  {.importcpp: "forward".}

proc reset*(this: var HingeEmbeddingLossImpl)  {.importcpp: "reset".}

proc pretty_print*(this: HingeEmbeddingLossImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `HingeEmbeddingLoss` module into the given `stream`.

proc forward*(this: var HingeEmbeddingLossImpl): int  {.importcpp: "forward".}

proc reset*(this: var MultiMarginLossImpl)  {.importcpp: "reset".}

proc pretty_print*(this: MultiMarginLossImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `MultiMarginLoss` module into the given `stream`.

proc forward*(this: var MultiMarginLossImpl): int  {.importcpp: "forward".}

proc reset*(this: var CosineEmbeddingLossImpl)  {.importcpp: "reset".}

proc pretty_print*(this: CosineEmbeddingLossImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `CosineEmbeddingLoss` module into the given
  ## `stream`.

proc forward*(this: var CosineEmbeddingLossImpl): int  {.importcpp: "forward".}

proc reset*(this: var SmoothL1LossImpl)  {.importcpp: "reset".}

proc pretty_print*(this: SmoothL1LossImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `L1Loss` module into the given `stream`.

proc forward*(this: var SmoothL1LossImpl): int  {.importcpp: "forward".}

proc reset*(this: var MultiLabelMarginLossImpl)  {.importcpp: "reset".}

proc pretty_print*(this: MultiLabelMarginLossImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `L1Loss` module into the given `stream`.

proc forward*(this: var MultiLabelMarginLossImpl): int  {.importcpp: "forward".}

proc pretty_print*(this: SoftMarginLossImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `SoftMarginLoss` module into the given `stream`.

proc reset*(this: var SoftMarginLossImpl)  {.importcpp: "reset".}

proc forward*(this: var SoftMarginLossImpl): int  {.importcpp: "forward".}

proc pretty_print*(this: MultiLabelSoftMarginLossImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `MultiLabelSoftMarginLoss` module into the given
  ## `stream`.

proc reset*(this: var MultiLabelSoftMarginLossImpl)  {.importcpp: "reset".}

proc forward*(this: var MultiLabelSoftMarginLossImpl): int  {.importcpp: "forward".}

proc reset*(this: var TripletMarginLossImpl)  {.importcpp: "reset".}

proc pretty_print*(this: TripletMarginLossImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `TripletMarginLoss` module into the given `stream`.

proc forward*(this: var TripletMarginLossImpl): int  {.importcpp: "forward".}

proc reset*(this: var CTCLossImpl)  {.importcpp: "reset".}

proc pretty_print*(this: CTCLossImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `CTCLoss` module into the given `stream`.

proc forward*(this: var CTCLossImpl): int  {.importcpp: "forward".}

proc reset*(this: var PoissonNLLLossImpl)  {.importcpp: "reset".}

proc pretty_print*(this: PoissonNLLLossImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `PoissonNLLLoss` module into the given `stream`.

proc forward*(this: var PoissonNLLLossImpl): int  {.importcpp: "forward".}

proc reset*(this: var MarginRankingLossImpl)  {.importcpp: "reset".}

proc pretty_print*(this: MarginRankingLossImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `MarginRankingLoss` module into the given `stream`.

proc forward*(this: var MarginRankingLossImpl): int  {.importcpp: "forward".}

proc pretty_print*(this: NLLLossImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `NLLLoss` module into the given `stream`.

proc reset*(this: var NLLLossImpl)  {.importcpp: "reset".}

proc forward*(this: var NLLLossImpl): int  {.importcpp: "forward".}

proc reset*(this: var CrossEntropyLossImpl)  {.importcpp: "reset".}

proc pretty_print*(this: CrossEntropyLossImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `CrossEntropyLoss` module into the given `stream`.

proc forward*(this: var CrossEntropyLossImpl): int  {.importcpp: "forward".}

proc reset*(this: var BCEWithLogitsLossImpl)  {.importcpp: "reset".}

proc pretty_print*(this: BCEWithLogitsLossImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `BCEWithLogitsLoss` module into the given `stream`.

proc forward*(this: var BCEWithLogitsLossImpl): int  {.importcpp: "forward".}

{.pop.} # header: "nn/modules/loss.h
