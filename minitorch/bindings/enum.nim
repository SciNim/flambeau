{.push header: "enum.h".}


# Constructors and methods
proc constructor_kLinear*(): kLinear {.constructor,importcpp: "kLinear".}

proc constructor_kConv1D*(): kConv1D {.constructor,importcpp: "kConv1D".}

proc constructor_kConv2D*(): kConv2D {.constructor,importcpp: "kConv2D".}

proc constructor_kConv3D*(): kConv3D {.constructor,importcpp: "kConv3D".}

proc constructor_kConvTranspose1D*(): kConvTranspose1D {.constructor,importcpp: "kConvTranspose1D".}

proc constructor_kConvTranspose2D*(): kConvTranspose2D {.constructor,importcpp: "kConvTranspose2D".}

proc constructor_kConvTranspose3D*(): kConvTranspose3D {.constructor,importcpp: "kConvTranspose3D".}

proc constructor_kSigmoid*(): kSigmoid {.constructor,importcpp: "kSigmoid".}

proc constructor_kTanh*(): kTanh {.constructor,importcpp: "kTanh".}

proc constructor_kReLU*(): kReLU {.constructor,importcpp: "kReLU".}

proc constructor_kGELU*(): kGELU {.constructor,importcpp: "kGELU".}

proc constructor_kSiLU*(): kSiLU {.constructor,importcpp: "kSiLU".}

proc constructor_kLeakyReLU*(): kLeakyReLU {.constructor,importcpp: "kLeakyReLU".}

proc constructor_kFanIn*(): kFanIn {.constructor,importcpp: "kFanIn".}

proc constructor_kFanOut*(): kFanOut {.constructor,importcpp: "kFanOut".}

proc constructor_kConstant*(): kConstant {.constructor,importcpp: "kConstant".}

proc constructor_kReflect*(): kReflect {.constructor,importcpp: "kReflect".}

proc constructor_kReplicate*(): kReplicate {.constructor,importcpp: "kReplicate".}

proc constructor_kCircular*(): kCircular {.constructor,importcpp: "kCircular".}

proc constructor_kNearest*(): kNearest {.constructor,importcpp: "kNearest".}

proc constructor_kBilinear*(): kBilinear {.constructor,importcpp: "kBilinear".}

proc constructor_kBicubic*(): kBicubic {.constructor,importcpp: "kBicubic".}

proc constructor_kTrilinear*(): kTrilinear {.constructor,importcpp: "kTrilinear".}

proc constructor_kArea*(): kArea {.constructor,importcpp: "kArea".}

proc constructor_kSum*(): kSum {.constructor,importcpp: "kSum".}

proc constructor_kMean*(): kMean {.constructor,importcpp: "kMean".}

proc constructor_kMax*(): kMax {.constructor,importcpp: "kMax".}

proc constructor_kNone*(): kNone {.constructor,importcpp: "kNone".}

proc constructor_kBatchMean*(): kBatchMean {.constructor,importcpp: "kBatchMean".}

proc constructor_kZeros*(): kZeros {.constructor,importcpp: "kZeros".}

proc constructor_kBorder*(): kBorder {.constructor,importcpp: "kBorder".}

proc constructor_kReflection*(): kReflection {.constructor,importcpp: "kReflection".}

proc constructor_kRNN_TANH*(): kRNN_TANH {.constructor,importcpp: "kRNN_TANH".}

proc constructor_kRNN_RELU*(): kRNN_RELU {.constructor,importcpp: "kRNN_RELU".}

proc constructor_kLSTM*(): kLSTM {.constructor,importcpp: "kLSTM".}

proc constructor_kGRU*(): kGRU {.constructor,importcpp: "kGRU".}

proc `()`*(this: _compute_enum_name, v: enumtype::kLinear): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kConv1D): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kConv2D): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kConv3D): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kConvTranspose1D): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kConvTranspose2D): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kConvTranspose3D): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kSigmoid): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kTanh): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kReLU): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kGELU): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kSiLU): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kLeakyReLU): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kFanIn): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kFanOut): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kConstant): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kReflect): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kReplicate): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kCircular): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kNearest): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kBilinear): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kBicubic): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kTrilinear): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kArea): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kSum): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kMean): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kMax): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kNone): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kBatchMean): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kZeros): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kBorder): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kReflection): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kRNN_TANH): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kRNN_RELU): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kLSTM): std::string  {.importcpp: "`()`".}

proc `()`*(this: _compute_enum_name, v: enumtype::kGRU): std::string  {.importcpp: "`()`".}

{.pop.} # header: "enum.h
