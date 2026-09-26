#![no_std]
#![cfg_attr(feature = "const-trait", feature(const_trait_impl))]

//! Reshaping of arrays, by value, by reference and by mutable reference, all in `const fn`.
//!
//! [`<[[T; N]]>::as_flattened()`](slice::as_flattened) and
//! [`<[T]>::as_chunks()`](slice::as_chunks) in the standard library convert to and from slices,
//! losing the length in the process. This crate converts between arrays of the same total length
//! instead, for example `[[T; N]; M]` into `[T; N * M]`:
//!
//! ```
//! use array_reshape::{Flatten, Rechunk, Unflatten};
//!
//! let pairs = [[1, 2], [3, 4], [5, 6]];
//!
//! let flat: &[u8; 6] = pairs.flatten_ref();
//! assert_eq!(flat, &[1, 2, 3, 4, 5, 6]);
//!
//! let triples = pairs.rechunk_ref::<3, 2>();
//! assert_eq!(triples, &[[1, 2, 3], [4, 5, 6]]);
//!
//! let pairs_again = flat.unflatten::<2, 3>();
//! assert_eq!(pairs_again, pairs);
//! ```
//!
//! Slices of arrays can be reshaped element by element too:
//!
//! ```
//! use array_reshape::{FlattenEach, UnflattenEach};
//!
//! let bytes = [[1, 2, 3, 4], [5, 6, 7, 8]];
//!
//! let pairs = bytes.unflatten_each_ref::<2, 2>();
//! assert_eq!(pairs, &[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
//!
//! let bytes_again: &[[u8; 4]] = pairs.flatten_each_ref();
//! assert_eq!(bytes_again, &bytes);
//! ```
//!
//! # `const fn` support
//!
//! Trait methods can't be called in `const fn` on stable Rust, which is why every conversion is
//! also available as a free `const fn` with the same name:
//!
//! ```
//! const FLAT: [u8; 4] = array_reshape::flatten([[1, 2], [3, 4]]);
//! assert_eq!(FLAT, [1, 2, 3, 4]);
//! ```
//!
//! With the `const-trait` feature, which requires nightly Rust, all traits become `const trait`s,
//! so their methods can be called in `const fn` directly. Crates calling them in `const fn` need to
//! enable `const_trait_impl` feature themselves.
//!
//! # Output length
//!
//! Stable Rust doesn't allow computing array lengths from generic constants yet (which would
//! require `generic_const_exprs` feature), so the length of the output is a separate generic
//! parameter, which is usually inferred from the context. The lengths are checked at compile
//! time, but since the check happens after monomorphization, it is only reported by
//! `cargo build`, not `cargo check`:
//!
//! ```compile_fail
//! use array_reshape::Flatten;
//!
//! let pairs = [[1, 2], [3, 4], [5, 6]];
//!
//! let flat: &[u8; 5] = pairs.flatten_ref();
//! ```
//!
//! The check also happens for code that is never executed, but is compiled, like `match` arms
//! that don't match for a particular set of generic parameters. Convert the input into an array
//! of a concrete size in such cases first (for example with
//! [`<[T]>::as_array()`](slice::as_array)).

use core::mem::ManuallyDrop;
use core::{mem, ptr, slice};

/// Checks at compile time that `inner * outer == len`
#[inline(always)]
const fn assert_len<const INNER: usize, const OUTER: usize, const LEN: usize>() {
    const {
        assert!(
            matches!(INNER.checked_mul(OUTER), Some(len) if len == LEN),
            "Arrays must have the same total number of elements"
        );
    }
}

/// Moves `src` into `Dst` of the same size and layout
///
/// # Safety
/// `Src` and `Dst` must be arrays of the same element type with the same total number of elements.
#[inline(always)]
const unsafe fn transmute_array<Src, Dst>(src: Src) -> Dst {
    let src = ManuallyDrop::new(src);
    // SAFETY: Guaranteed by function contract, the original value is not dropped
    unsafe { mem::transmute_copy::<ManuallyDrop<Src>, Dst>(&src) }
}

/// Flatten `[[T; N]; M]` into `[T; K]`, where `K == N * M`
#[inline(always)]
pub const fn flatten<T, const N: usize, const M: usize, const K: usize>(
    array: [[T; N]; M],
) -> [T; K] {
    assert_len::<N, M, K>();
    // SAFETY: The same element type and the same number of elements, checked above
    unsafe { transmute_array(array) }
}

/// Flatten `&[[T; N]; M]` into `&[T; K]`, where `K == N * M`
#[inline(always)]
pub const fn flatten_ref<T, const N: usize, const M: usize, const K: usize>(
    array: &[[T; N]; M],
) -> &[T; K] {
    assert_len::<N, M, K>();
    // SAFETY: The same element type and the same number of elements, checked above
    unsafe { &*ptr::from_ref(array).cast::<[T; K]>() }
}

/// Flatten `&mut [[T; N]; M]` into `&mut [T; K]`, where `K == N * M`
#[inline(always)]
pub const fn flatten_mut<T, const N: usize, const M: usize, const K: usize>(
    array: &mut [[T; N]; M],
) -> &mut [T; K] {
    assert_len::<N, M, K>();
    // SAFETY: The same element type and the same number of elements, checked above
    unsafe { &mut *ptr::from_mut(array).cast::<[T; K]>() }
}

/// Unflatten `[T; K]` into `[[T; N]; M]`, where `K == N * M`
#[inline(always)]
pub const fn unflatten<T, const N: usize, const M: usize, const K: usize>(
    array: [T; K],
) -> [[T; N]; M] {
    assert_len::<N, M, K>();
    // SAFETY: The same element type and the same number of elements, checked above
    unsafe { transmute_array(array) }
}

/// Unflatten `&[T; K]` into `&[[T; N]; M]`, where `K == N * M`
#[inline(always)]
pub const fn unflatten_ref<T, const N: usize, const M: usize, const K: usize>(
    array: &[T; K],
) -> &[[T; N]; M] {
    assert_len::<N, M, K>();
    // SAFETY: The same element type and the same number of elements, checked above
    unsafe { &*ptr::from_ref(array).cast::<[[T; N]; M]>() }
}

/// Unflatten `&mut [T; K]` into `&mut [[T; N]; M]`, where `K == N * M`
#[inline(always)]
pub const fn unflatten_mut<T, const N: usize, const M: usize, const K: usize>(
    array: &mut [T; K],
) -> &mut [[T; N]; M] {
    assert_len::<N, M, K>();
    // SAFETY: The same element type and the same number of elements, checked above
    unsafe { &mut *ptr::from_mut(array).cast::<[[T; N]; M]>() }
}

/// Change chunk size of `[[T; A]; B]` to get `[[T; C]; D]`, where `A * B == C * D`
#[inline(always)]
pub const fn rechunk<T, const A: usize, const B: usize, const C: usize, const D: usize>(
    array: [[T; A]; B],
) -> [[T; C]; D] {
    assert_same_total_len::<A, B, C, D>();
    // SAFETY: The same element type and the same number of elements, checked above
    unsafe { transmute_array(array) }
}

/// Change chunk size of `&[[T; A]; B]` to get `&[[T; C]; D]`, where `A * B == C * D`
#[inline(always)]
pub const fn rechunk_ref<T, const A: usize, const B: usize, const C: usize, const D: usize>(
    array: &[[T; A]; B],
) -> &[[T; C]; D] {
    assert_same_total_len::<A, B, C, D>();
    // SAFETY: The same element type and the same number of elements, checked above
    unsafe { &*ptr::from_ref(array).cast::<[[T; C]; D]>() }
}

/// Change chunk size of `&mut [[T; A]; B]` to get `&mut [[T; C]; D]`, where `A * B == C * D`
#[inline(always)]
pub const fn rechunk_mut<T, const A: usize, const B: usize, const C: usize, const D: usize>(
    array: &mut [[T; A]; B],
) -> &mut [[T; C]; D] {
    assert_same_total_len::<A, B, C, D>();
    // SAFETY: The same element type and the same number of elements, checked above
    unsafe { &mut *ptr::from_mut(array).cast::<[[T; C]; D]>() }
}

/// Flatten each element of `&[[[T; N]; M]]` to get `&[[T; K]]`, where `K == N * M`
#[inline(always)]
pub const fn flatten_each_ref<T, const N: usize, const M: usize, const K: usize>(
    slice: &[[[T; N]; M]],
) -> &[[T; K]] {
    assert_len::<N, M, K>();
    // SAFETY: The same element type and the same number of elements in each slice element, checked
    // above
    unsafe { slice::from_raw_parts(slice.as_ptr().cast::<[T; K]>(), slice.len()) }
}

/// Flatten each element of `&mut [[[T; N]; M]]` to get `&mut [[T; K]]`, where `K == N * M`
#[inline(always)]
pub const fn flatten_each_mut<T, const N: usize, const M: usize, const K: usize>(
    slice: &mut [[[T; N]; M]],
) -> &mut [[T; K]] {
    assert_len::<N, M, K>();
    // SAFETY: The same element type and the same number of elements in each slice element, checked
    // above
    unsafe { slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<[T; K]>(), slice.len()) }
}

/// Unflatten each element of `&[[T; K]]` to get `&[[[T; N]; M]]`, where `K == N * M`
#[inline(always)]
pub const fn unflatten_each_ref<T, const N: usize, const M: usize, const K: usize>(
    slice: &[[T; K]],
) -> &[[[T; N]; M]] {
    assert_len::<N, M, K>();
    // SAFETY: The same element type and the same number of elements in each slice element, checked
    // above
    unsafe { slice::from_raw_parts(slice.as_ptr().cast::<[[T; N]; M]>(), slice.len()) }
}

/// Unflatten each element of `&mut [[T; K]]` to get `&mut [[[T; N]; M]]`, where `K == N * M`
#[inline(always)]
pub const fn unflatten_each_mut<T, const N: usize, const M: usize, const K: usize>(
    slice: &mut [[T; K]],
) -> &mut [[[T; N]; M]] {
    assert_len::<N, M, K>();
    // SAFETY: The same element type and the same number of elements in each slice element, checked
    // above
    unsafe { slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<[[T; N]; M]>(), slice.len()) }
}

/// Change chunk size of each element of `&[[[T; A]; B]]` to get `&[[[T; C]; D]]`, where
/// `A * B == C * D`
#[inline(always)]
pub const fn rechunk_each_ref<T, const A: usize, const B: usize, const C: usize, const D: usize>(
    slice: &[[[T; A]; B]],
) -> &[[[T; C]; D]] {
    assert_same_total_len::<A, B, C, D>();
    // SAFETY: The same element type and the same number of elements in each slice element, checked
    // above
    unsafe { slice::from_raw_parts(slice.as_ptr().cast::<[[T; C]; D]>(), slice.len()) }
}

/// Change chunk size of each element of `&mut [[[T; A]; B]]` to get `&mut [[[T; C]; D]]`, where
/// `A * B == C * D`
#[inline(always)]
pub const fn rechunk_each_mut<T, const A: usize, const B: usize, const C: usize, const D: usize>(
    slice: &mut [[[T; A]; B]],
) -> &mut [[[T; C]; D]] {
    assert_same_total_len::<A, B, C, D>();
    // SAFETY: The same element type and the same number of elements in each slice element, checked
    // above
    unsafe { slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<[[T; C]; D]>(), slice.len()) }
}

/// Checks at compile time that `a * b == c * d`
#[inline(always)]
const fn assert_same_total_len<const A: usize, const B: usize, const C: usize, const D: usize>() {
    const {
        assert!(
            matches!(
                (A.checked_mul(B), C.checked_mul(D)),
                (Some(input_len), Some(output_len)) if input_len == output_len
            ),
            "Arrays must have the same total number of elements"
        );
    }
}

/// Defines reshaping traits, optionally as `const trait`s.
///
/// This is a macro because `const trait` syntax is rejected by stable Rust even in code that is
/// disabled with `#[cfg]`, but not in unexpanded macro invocations.
macro_rules! define_traits {
    ($($const:ident)?) => {
        /// Reshaping of `[[T; N]; M]`, see crate-level documentation for examples.
        ///
        /// The same conversions are available as free `const fn`s, see [`flatten()`] and others.
        pub $($const)? trait Flatten<T, const N: usize, const M: usize>: Sized {
            /// Flatten into `[T; K]`, where `K == N * M`
            fn flatten<const K: usize>(self) -> [T; K];

            /// Flatten into `&[T; K]`, where `K == N * M`
            fn flatten_ref<const K: usize>(&self) -> &[T; K];

            /// Flatten into `&mut [T; K]`, where `K == N * M`
            fn flatten_mut<const K: usize>(&mut self) -> &mut [T; K];
        }

        $($const)? impl<T, const N: usize, const M: usize> Flatten<T, N, M> for [[T; N]; M] {
            #[inline(always)]
            fn flatten<const K: usize>(self) -> [T; K] {
                flatten(self)
            }

            #[inline(always)]
            fn flatten_ref<const K: usize>(&self) -> &[T; K] {
                flatten_ref(self)
            }

            #[inline(always)]
            fn flatten_mut<const K: usize>(&mut self) -> &mut [T; K] {
                flatten_mut(self)
            }
        }

        /// Changing chunk size of `[[T; N]; M]`, see crate-level documentation for examples.
        ///
        /// The same conversions are available as free `const fn`s, see [`rechunk()`] and others.
        pub $($const)? trait Rechunk<T, const N: usize, const M: usize>: Sized {
            /// Change chunk size to get `[[T; C]; D]`, where `N * M == C * D`
            fn rechunk<const C: usize, const D: usize>(self) -> [[T; C]; D];

            /// Change chunk size to get `&[[T; C]; D]`, where `N * M == C * D`
            fn rechunk_ref<const C: usize, const D: usize>(&self) -> &[[T; C]; D];

            /// Change chunk size to get `&mut [[T; C]; D]`, where `N * M == C * D`
            fn rechunk_mut<const C: usize, const D: usize>(&mut self) -> &mut [[T; C]; D];
        }

        $($const)? impl<T, const N: usize, const M: usize> Rechunk<T, N, M> for [[T; N]; M] {
            #[inline(always)]
            fn rechunk<const C: usize, const D: usize>(self) -> [[T; C]; D] {
                rechunk(self)
            }

            #[inline(always)]
            fn rechunk_ref<const C: usize, const D: usize>(&self) -> &[[T; C]; D] {
                rechunk_ref(self)
            }

            #[inline(always)]
            fn rechunk_mut<const C: usize, const D: usize>(&mut self) -> &mut [[T; C]; D] {
                rechunk_mut(self)
            }
        }

        /// Reshaping of `[T; K]`, see crate-level documentation for examples.
        ///
        /// The same conversions are available as free `const fn`s, see [`unflatten()`] and others.
        pub $($const)? trait Unflatten<T, const K: usize>: Sized {
            /// Unflatten into `[[T; N]; M]`, where `K == N * M`
            fn unflatten<const N: usize, const M: usize>(self) -> [[T; N]; M];

            /// Unflatten into `&[[T; N]; M]`, where `K == N * M`
            fn unflatten_ref<const N: usize, const M: usize>(&self) -> &[[T; N]; M];

            /// Unflatten into `&mut [[T; N]; M]`, where `K == N * M`
            fn unflatten_mut<const N: usize, const M: usize>(&mut self) -> &mut [[T; N]; M];
        }

        $($const)? impl<T, const K: usize> Unflatten<T, K> for [T; K] {
            #[inline(always)]
            fn unflatten<const N: usize, const M: usize>(self) -> [[T; N]; M] {
                unflatten(self)
            }

            #[inline(always)]
            fn unflatten_ref<const N: usize, const M: usize>(&self) -> &[[T; N]; M] {
                unflatten_ref(self)
            }

            #[inline(always)]
            fn unflatten_mut<const N: usize, const M: usize>(&mut self) -> &mut [[T; N]; M] {
                unflatten_mut(self)
            }
        }
        /// Flattening of each element of `[[[T; N]; M]]`, see crate-level documentation for examples.
        ///
        /// The same conversions are available as free `const fn`s, see [`flatten_each_ref()`] and
        /// [`flatten_each_mut()`].
        pub $($const)? trait FlattenEach<T, const N: usize, const M: usize> {
            /// Flatten each element to get `&[[T; K]]`, where `K == N * M`
            fn flatten_each_ref<const K: usize>(&self) -> &[[T; K]];

            /// Flatten each element to get `&mut [[T; K]]`, where `K == N * M`
            fn flatten_each_mut<const K: usize>(&mut self) -> &mut [[T; K]];
        }

        $($const)? impl<T, const N: usize, const M: usize> FlattenEach<T, N, M> for [[[T; N]; M]] {
            #[inline(always)]
            fn flatten_each_ref<const K: usize>(&self) -> &[[T; K]] {
                flatten_each_ref(self)
            }

            #[inline(always)]
            fn flatten_each_mut<const K: usize>(&mut self) -> &mut [[T; K]] {
                flatten_each_mut(self)
            }
        }

        /// Unflattening of each element of `[[T; K]]`, see crate-level documentation for examples.
        ///
        /// The same conversions are available as free `const fn`s, see [`unflatten_each_ref()`]
        /// and [`unflatten_each_mut()`].
        pub $($const)? trait UnflattenEach<T, const K: usize> {
            /// Unflatten each element to get `&[[[T; N]; M]]`, where `K == N * M`
            fn unflatten_each_ref<const N: usize, const M: usize>(&self) -> &[[[T; N]; M]];

            /// Unflatten each element to get `&mut [[[T; N]; M]]`, where `K == N * M`
            fn unflatten_each_mut<const N: usize, const M: usize>(&mut self) -> &mut [[[T; N]; M]];
        }

        $($const)? impl<T, const K: usize> UnflattenEach<T, K> for [[T; K]] {
            #[inline(always)]
            fn unflatten_each_ref<const N: usize, const M: usize>(&self) -> &[[[T; N]; M]] {
                unflatten_each_ref(self)
            }

            #[inline(always)]
            fn unflatten_each_mut<const N: usize, const M: usize>(&mut self) -> &mut [[[T; N]; M]] {
                unflatten_each_mut(self)
            }
        }

        /// Changing chunk size of each element of `[[[T; N]; M]]`, see crate-level documentation
        /// for examples.
        ///
        /// The same conversions are available as free `const fn`s, see [`rechunk_each_ref()`] and
        /// [`rechunk_each_mut()`].
        pub $($const)? trait RechunkEach<T, const N: usize, const M: usize> {
            /// Change chunk size of each element to get `&[[[T; C]; D]]`, where `N * M == C * D`
            fn rechunk_each_ref<const C: usize, const D: usize>(&self) -> &[[[T; C]; D]];

            /// Change chunk size of each element to get `&mut [[[T; C]; D]]`, where
            /// `N * M == C * D`
            fn rechunk_each_mut<const C: usize, const D: usize>(&mut self) -> &mut [[[T; C]; D]];
        }

        $($const)? impl<T, const N: usize, const M: usize> RechunkEach<T, N, M> for [[[T; N]; M]] {
            #[inline(always)]
            fn rechunk_each_ref<const C: usize, const D: usize>(&self) -> &[[[T; C]; D]] {
                rechunk_each_ref(self)
            }

            #[inline(always)]
            fn rechunk_each_mut<const C: usize, const D: usize>(&mut self) -> &mut [[[T; C]; D]] {
                rechunk_each_mut(self)
            }
        }
    };
}

#[cfg(feature = "const-trait")]
define_traits!(const);
#[cfg(not(feature = "const-trait"))]
define_traits!();
