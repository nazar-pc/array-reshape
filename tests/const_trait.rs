//! Tests for `const-trait` feature, which makes methods callable in `const fn`

#![cfg(feature = "const-trait")]
#![cfg_attr(feature = "const-trait", feature(const_trait_impl))]

use array_reshape::{Flatten, Rechunk, Unflatten};

#[test]
fn methods_in_const_fn() {
    const FLAT: [u8; 4] = [[1, 2], [3, 4]].flatten();
    const FLAT_REF: &[u8; 4] = [[1, 2], [3, 4]].flatten_ref();
    const CHUNKS: [[u8; 2]; 2] = [1, 2, 3, 4].unflatten();
    const RECHUNKED: [[u8; 1]; 4] = [[1, 2], [3, 4]].rechunk();
    const MODIFIED: [u8; 4] = {
        let mut array = [1, 2, 3, 4];
        array.unflatten_mut::<2, 2>()[1][0] = 5;
        array.unflatten_mut::<4, 1>()[0][1] = 6;
        array
    };

    assert_eq!(FLAT, [1, 2, 3, 4]);
    assert_eq!(FLAT_REF, &[1, 2, 3, 4]);
    assert_eq!(CHUNKS, [[1, 2], [3, 4]]);
    assert_eq!(RECHUNKED, [[1], [2], [3], [4]]);
    assert_eq!(MODIFIED, [1, 6, 5, 4]);
}
