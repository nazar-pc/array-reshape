# array-reshape

[![Crates.io](https://img.shields.io/crates/v/array-reshape.svg)](https://crates.io/crates/array-reshape)
[![Docs](https://docs.rs/array-reshape/badge.svg)](https://docs.rs/array-reshape)

Reshaping of arrays, by value, by reference and by mutable reference, all in `const fn`.

`<[[T; N]]>::as_flattened()` and `<[T]>::as_chunks()` in the standard library convert to and from slices, losing the
length in the process. This crate converts between arrays of the same total length instead, for example `[[T; N]; M]`
into `[T; N * M]`:

```rust
use array_reshape::{Flatten, Rechunk, Unflatten};

let pairs = [[1, 2], [3, 4], [5, 6]];

let flat: &[u8; 6] = pairs.flatten_ref();
assert_eq!(flat, &[1, 2, 3, 4, 5, 6]);

let triples = pairs.rechunk_ref::<3, 2>();
assert_eq!(triples, &[[1, 2, 3], [4, 5, 6]]);

let pairs_again = flat.unflatten::<2, 3>();
assert_eq!(pairs_again, pairs);
```

Every conversion is also available as a free `const fn` (like `array_reshape::flatten_ref()`) that works in `const`
context on stable Rust. With `const-trait` feature, which requires nightly Rust, `Flatten`, `Rechunk` and `Unflatten`
become `const trait`s, so their methods can be called in `const fn` directly.

Stable Rust doesn't allow computing array lengths from generic constants yet, so the length of the output is a separate
generic parameter, which is usually inferred from the context. The lengths are checked at compile time, but since the
check happens after monomorphization, it is only reported by `cargo build`, not `cargo check`.

The crate is `no_std` and works on stable Rust 1.85 or newer.
