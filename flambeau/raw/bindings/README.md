# Raw bindings to LibTorch

This provides almost raw bindings to PyTorch tensors.

"Nimification" (camelCase), ergonomic indexing and interoperability with Nim types is left to the "high-level" bindings.
This should ease searching PyTorch and libtorch documentation,
and make C++ tutorials easily applicable.

Nonetheless some slight modifications were given to the raw bindings:
- `&=`, `|=` and `^=` have been renamed bitand, bitor, bitxor
- `[]` and `[]=` are not exported as index and index_put are more flexible
  and we want to leave those symbols available for Numpy-like ergonomic indexing.
- Nim's `index_fill_mut` and `masked_fill_mut` are mapped to the in-place
  C++ `index_fill_` and `masked_fill_`.
  The original out-of-place versions are doing clone+in-place mutation

## Adding new bindings

### Conventions

PyTorch exposes out-of-place and in-place versions of functions.

The out-of-place version is usually named `foo` and in-place `foo_`.
In Nim we use `foo` and `foo_mut`.

Also the mutating version returns a reference to the mutated tensors. This is something that we avoid in Nim and so mutated version should return nothing.

### Tensors

The tensor definitions are stored in 2 locations:

1. Methods where the first argument is an implicit tensor are in `libtorch/include/ATen/core/TensorBody.h`
2. Free-standing functions that are not associated with the tensor class
   are in `libtorch/include/ATen/Functions.h`
3. Some functions exist both as methods or as free-standing.
   In Nim there is no distinction. It is preferred to wrap
   directly the "method" functions as in PyTorch upstream,
   the free-standing functions just call the method functions underneath,
   so there is no need to add yet another indirection.

We use the namespace `torch::Foo` instead of the low-level `at::Foo`
as this is the one recommended for C++ users.

API reference:
- Tensor methods: https://pytorch.org/cppdocs/api/classat_1_1_tensor.html
- Free-standing functions: https://pytorch.org/cppdocs/api/file_build_aten_src_ATen_Functions.h.html

#### Dealing with c10::optional

c10::optional is similar to Nim `Option` for PyTorch.
The type `T` is implicitly convertible to `c10::optional<T>` at the C++ compiler level
hence we don't expose `c10::optional` in Nim but the bindings directly use the base type and an overload with no input.

#### On TensorOptions, Device, DeviceKind and ScalarType

`TensorOptions` has a default param of empty `{}`.
Also DeviceKind (for example kCuda), Device ({kCuda, 0} for Cuda GPU0 and scalar type (kFloat32) are implictly convertible to TensorOptions.

This means that for ergonomic use, it's best to create overload with each parameters, for instance for

```C++
Tensor at::eye(int64_t n, const TensorOptions &options = {})
```

We translate that into
```Nim
func eye*(n: int64): Tensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, options: TensorOptions): Tensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, scalarKind: ScalarKind): Tensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, device: DeviceKind): Tensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, device: Device): Tensor {.importcpp: "torch::eye(@)".}
```

## Passing or skipping types or static integer

Nim is able to pass types to C++ with the following base behavior:

If the type is attached to an argument or the return value, use the syntax
  - `'0` to get the type of the return value
  - `'1` to get the type of the first argument
  - `i*0` to get the subtype of the generic return value, for example to extract T from CppVector[T]

### Typedesc or static arguments

Typedesc or static argument are skipped **in the codegen** when used with `#` or `@`
Note that for counting purposes, typedesc or static argument still increment the count.

For example for tuples, this will properly extract the tuple value

```Nim
type
  CppTuple2* {.importcpp: "std::tuple".} [T0, T1] = object
  CppTuple3* {.importcpp: "std::tuple".} [T0, T1, T2] = object
  CppTuple4* {.importcpp: "std::tuple".} [T0, T1, T2, T3] = object
  CppTuple5* {.importcpp: "std::tuple".} [T0, T1, T2, T3, T4] = object

  CppTuple = CppTuple2|CppTuple3|CppTuple4|CppTuple5

func tupGet(index: int, tup: CppTuple, outT: type): outT {.importcpp: "std::get<#>(#)".}
  ## C++ get from tuple.
  ## We have to use this unnatural argument order at low-level
  ## and add an outType parameter for out type inference
```

The `index` must be an `int` and not a `static int` or it would be skipped.
The `outT: type` would be skipped even if we use `@`.
To finish wrapping tuple we need an extra `get` procedure which will give us
a more intuitive ordering.
```Nim
template get*(tup: CppTuple, index: static int): auto =
```

### Constructors with typedesc

This behavior allow us to wrap constructor like this
```Nim
func init*(T: type SGDOptions, learning_rate: float64): T {.constructor, importcpp: "torch::optim::SGDOptions(@)".}
```

The typedesc will not be passed to `@`

### Passing typedesc to C++

In some cases we might want to pass typedesc to C++, instead of leaving them
just in Nim.
Since they are skipped with `#` or `@` we need to pass them with the `'1` syntax

This is done the following way:

```Nim
func make_data_loader*[D: BatchDataset; S: Sampler](
       SamplerType: type S,
       dataset: D,
       batch_size: csize_t
  ): CppUniquePtr[StatelessDataLoader[D, SamplerType]] {.
  importcpp: "torch::data::make_data_loader<'*1>(@)".}
```

A call with `make_data_loader(SequentialSampler, dataset, 64)` will produce
```C++
torch::data::make_data_loader<torch::data::Sampler:SequentialSampler>(dataset, 64)
```

### Avoiding Nim temporaries

For codegen Nim generates temporaries but this assumes that the temporary types has a default constructor.

To avoid that we can tag the importcpp proc as `{.noInit.}

This is especially important for dereferencing operators.
