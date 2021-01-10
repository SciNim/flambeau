// We need Linear{nullptr} in the codegen
// so we would like to cheat by inlining C++
//
// type Net {.pure.} = object of Module
//
//   fc1: Linear
//   fc2: Linear
//   fc3: Linear
//
// https://github.com/nim-lang/Nim/issues/4687
//
// However
// due to https://github.com/nim-lang/Nim/issues/16664
// it needs to be in its own file

#include <torch/torch.h>

struct Net: public torch::nn::Module {
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
  torch::nn::Linear fc3{nullptr};
};
