# 0.1.0

Initial release.

Features:

* `flatten()`, `flatten_ref()` and `flatten_mut()` for converting `[[T; N]; M]` into `[T; N * M]`
* `unflatten()`, `unflatten_ref()` and `unflatten_mut()` for converting `[T; N * M]` into `[[T; N]; M]`
* `rechunk()`, `rechunk_ref()` and `rechunk_mut()` for converting `[[T; A]; B]` into `[[T; C]; D]` with `A * B == C * D`
* All functions are `const fn` that work on stable Rust, with array lengths checked at compile time
* The same conversions as methods of `Flatten`, `Rechunk` and `Unflatten` traits implemented for arrays
* `const-trait` feature that makes `Flatten`, `Rechunk` and `Unflatten` `const trait`s on nightly Rust
