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
