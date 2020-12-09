{.push header: "detail/TensorDataContainer.h".}


# Constructors and methods
proc constructor_TensorDataContainer*(): TensorDataContainer {.constructor,importcpp: "TensorDataContainer".}

proc constructor_TensorDataContainer*(value: uint8_t): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(value: int8_t): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(value: int16_t): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(value: cint): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(value: int64_t): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(value: cfloat): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(value: cdouble): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(value: decltype(::c10::impl::ScalarTypeToCPPType< ::c10::ScalarType::Bool>::t)): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(value: decltype(::c10::impl::ScalarTypeToCPPType< ::c10::ScalarType::Half>::t)): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(value: decltype(::c10::impl::ScalarTypeToCPPType< ::c10::ScalarType::BFloat16>::t)): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(value: c10::complex<float>): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(value: c10::complex<double>): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(init_list: std::initializer_list<TensorDataContainer>): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: at::ArrayRef<uint8_t>): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: at::ArrayRef<int8_t>): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: at::ArrayRef<int16_t>): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: at::ArrayRef<int>): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: at::ArrayRef<int64_t>): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: at::ArrayRef<float>): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: at::ArrayRef<double>): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: at::ArrayRef<decltype(::c10::impl::ScalarTypeToCPPType< ::c10::ScalarType::Bool>::t)>): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: at::ArrayRef<decltype(::c10::impl::ScalarTypeToCPPType< ::c10::ScalarType::Half>::t)>): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: at::ArrayRef<decltype(::c10::impl::ScalarTypeToCPPType< ::c10::ScalarType::BFloat16>::t)>): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: at::ArrayRef<c10::complex<float>>): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: at::ArrayRef<c10::complex<double>>): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: cint): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: cint): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: cint): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: cint): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: cint): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: cint): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: cint): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: cint): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: cint): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: cint): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc constructor_TensorDataContainer*(values: cint): TensorDataContainer {.constructor,importcpp: "TensorDataContainer(@)".}

proc is_scalar*(this: TensorDataContainer): bool  {.importcpp: "is_scalar".}

proc scalar*(this: TensorDataContainer): c10::Scalar  {.importcpp: "scalar".}

proc is_init_list*(this: TensorDataContainer): bool  {.importcpp: "is_init_list".}

proc init_list*(this: TensorDataContainer): std::initializer_list<TensorDataContainer>  {.importcpp: "init_list".}

proc is_tensor*(this: TensorDataContainer): bool  {.importcpp: "is_tensor".}

proc tensor*(this: TensorDataContainer): at::Tensor  {.importcpp: "tensor".}

proc sizes*(this: TensorDataContainer): int  {.importcpp: "sizes".}

proc scalar_type*(this: TensorDataContainer): c10::ScalarType  {.importcpp: "scalar_type".}

proc convert_to_tensor*(this: TensorDataContainer, options: at::TensorOptions): at::Tensor  {.importcpp: "convert_to_tensor".}

proc pretty_print_recursive*(this: TensorDataContainer, stream: var std::ostream)  {.importcpp: "pretty_print_recursive".}

proc fill_tensor*(this: TensorDataContainer, tensor: var at::Tensor)  {.importcpp: "fill_tensor".}

{.pop.} # header: "detail/TensorDataContainer.h
