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
