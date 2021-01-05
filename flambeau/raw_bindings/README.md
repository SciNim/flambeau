# Raw bindings to LibTorch

This provides almost raw bindings to PyTorch tensors.

"Nimification" (camelCase), ergonomic indexing and interoperability with Nim types is left to the "high-level" bindings.
This should ease searching PyTorch and libtorch documentation,
and make C++ tutorials easily applicable.

Nonetheless some slight modifications were given to the raw bindings:
- `&=`, `|=` and `^=` have been renamed bitand, bitor, bitxor
- `[]` and `[]=` are not exported as index and index_put are more flexible
  and we want to leave those symbols available for Numpy-like ergonomic indexing.
- Nim's `index_fill` and `masked_fill` are mapped to the in-place
  C++ `index_fill_` and `masked_fill_`.
  The original out-of-place versions are doing clone+in-place mutation

## Adding new bindings

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
