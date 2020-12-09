{.push header: "nn/modules/distance.h".}


# Constructors and methods
proc constructor_CosineSimilarityImpl*(options_: cint): CosineSimilarityImpl {.constructor,importcpp: "CosineSimilarityImpl(@)".}

proc constructor_PairwiseDistanceImpl*(options_: cint): PairwiseDistanceImpl {.constructor,importcpp: "PairwiseDistanceImpl(@)".}

proc reset*(this: var CosineSimilarityImpl)  {.importcpp: "reset".}

proc pretty_print*(this: CosineSimilarityImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `CosineSimilarity` module into the given `stream`.

proc forward*(this: var CosineSimilarityImpl): int  {.importcpp: "forward".}

proc reset*(this: var PairwiseDistanceImpl)  {.importcpp: "reset".}

proc pretty_print*(this: PairwiseDistanceImpl, stream: cint)  {.importcpp: "pretty_print".}
  ## Pretty prints the `PairwiseDistance` module into the given `stream`.

proc forward*(this: var PairwiseDistanceImpl): int  {.importcpp: "forward".}

{.pop.} # header: "nn/modules/distance.h
