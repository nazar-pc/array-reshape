//! Tests for all reshaping functions and methods

use array_reshape::{Flatten, FlattenEach, Rechunk, RechunkEach, Unflatten, UnflattenEach};
use core::cell::Cell;
use core::marker::PhantomData;

#[test]
fn flatten() {
    let mut array = [[1_u16, 2, 3], [4, 5, 6]];

    let flat: [u16; 6] = array_reshape::flatten(array);
    assert_eq!(flat, [1, 2, 3, 4, 5, 6]);

    let flat: &[u16; 6] = array_reshape::flatten_ref(&array);
    assert_eq!(flat, &[1, 2, 3, 4, 5, 6]);

    let flat: &mut [u16; 6] = array_reshape::flatten_mut(&mut array);
    flat[3] = 7;
    assert_eq!(array, [[1, 2, 3], [7, 5, 6]]);
}

#[test]
fn unflatten() {
    let mut array = [1_u32, 2, 3, 4, 5, 6];

    let chunks: [[u32; 2]; 3] = array_reshape::unflatten(array);
    assert_eq!(chunks, [[1, 2], [3, 4], [5, 6]]);

    let chunks: &[[u32; 3]; 2] = array_reshape::unflatten_ref(&array);
    assert_eq!(chunks, &[[1, 2, 3], [4, 5, 6]]);

    let chunks: &mut [[u32; 1]; 6] = array_reshape::unflatten_mut(&mut array);
    chunks[5][0] = 7;
    assert_eq!(array, [1, 2, 3, 4, 5, 7]);
}

#[test]
fn rechunk() {
    let mut array = [[1_u64, 2], [3, 4], [5, 6]];

    let chunks: [[u64; 3]; 2] = array_reshape::rechunk(array);
    assert_eq!(chunks, [[1, 2, 3], [4, 5, 6]]);

    let chunks: &[[u64; 6]; 1] = array_reshape::rechunk_ref(&array);
    assert_eq!(chunks, &[[1, 2, 3, 4, 5, 6]]);

    let chunks: &mut [[u64; 3]; 2] = array_reshape::rechunk_mut(&mut array);
    chunks[1][0] = 7;
    assert_eq!(array, [[1, 2], [3, 7], [5, 6]]);
}

#[test]
fn empty_and_zero_sized() {
    let empty: [u8; 0] = array_reshape::flatten::<u8, 3, 0, 0>([]);
    assert_eq!(empty, []);
    let empty: [[u8; 0]; 5] = array_reshape::unflatten([]);
    assert_eq!(empty, [[]; 5]);

    let zero_sized = [[PhantomData::<u8>; 2]; 3];
    let flat: &[PhantomData<u8>; 6] = array_reshape::flatten_ref(&zero_sized);
    assert_eq!(flat.len(), 6);
    let rechunked: [[(); 3]; 2] = array_reshape::rechunk([[(); 2]; 3]);
    assert_eq!(rechunked, [[(); 3]; 2]);
}

#[test]
fn values_are_dropped_exactly_once() {
    struct DropCounter<'a>(&'a Cell<usize>, usize);

    impl Drop for DropCounter<'_> {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    let drops = Cell::new(0);
    let array = [
        [DropCounter(&drops, 0), DropCounter(&drops, 1)],
        [DropCounter(&drops, 2), DropCounter(&drops, 3)],
    ];

    let flat: [DropCounter<'_>; 4] = array_reshape::flatten(array);
    assert_eq!(drops.get(), 0);
    assert_eq!(flat.each_ref().map(|counter| counter.1), [0, 1, 2, 3]);

    let chunks: [[DropCounter<'_>; 1]; 4] = array_reshape::unflatten(flat);
    assert_eq!(drops.get(), 0);

    let chunks: [[DropCounter<'_>; 4]; 1] = array_reshape::rechunk(chunks);
    assert_eq!(drops.get(), 0);

    drop(chunks);
    assert_eq!(drops.get(), 4);
}

#[test]
fn const_fns() {
    const FLAT: [u8; 4] = array_reshape::flatten([[1, 2], [3, 4]]);
    const FLAT_REF: &[u8; 4] = array_reshape::flatten_ref(&[[1, 2], [3, 4]]);
    const CHUNKS: [[u8; 2]; 2] = array_reshape::unflatten([1, 2, 3, 4]);
    const CHUNKS_REF: &[[u8; 1]; 4] = array_reshape::unflatten_ref(&[1, 2, 3, 4]);
    const RECHUNKED: &[[u8; 4]; 1] = array_reshape::rechunk_ref(&[[1, 2], [3, 4]]);
    const MODIFIED: [[u8; 2]; 2] = {
        let mut array = [[1, 2], [3, 4]];
        array_reshape::flatten_mut::<_, 2, 2, 4>(&mut array)[1] = 5;
        array_reshape::rechunk_mut::<_, 2, 2, 4, 1>(&mut array)[0][2] = 6;
        array_reshape::unflatten_mut::<_, 1, 4, 4>(array_reshape::flatten_mut(&mut array))[3][0] =
            7;
        array
    };

    assert_eq!(FLAT, [1, 2, 3, 4]);
    assert_eq!(FLAT_REF, &[1, 2, 3, 4]);
    assert_eq!(CHUNKS, [[1, 2], [3, 4]]);
    assert_eq!(CHUNKS_REF, &[[1], [2], [3], [4]]);
    assert_eq!(RECHUNKED, &[[1, 2, 3, 4]]);
    assert_eq!(MODIFIED, [[1, 5], [6, 7]]);
}

#[test]
fn methods() {
    let mut array = [[1_u16, 2, 3], [4, 5, 6]];

    assert_eq!(array.flatten::<6>(), [1, 2, 3, 4, 5, 6]);
    assert_eq!(array.flatten_ref::<6>(), &[1, 2, 3, 4, 5, 6]);
    array.flatten_mut::<6>()[3] = 7;
    assert_eq!(array, [[1, 2, 3], [7, 5, 6]]);

    assert_eq!(array.rechunk::<2, 3>(), [[1, 2], [3, 7], [5, 6]]);
    assert_eq!(array.rechunk_ref::<6, 1>(), &[[1, 2, 3, 7, 5, 6]]);
    array.rechunk_mut::<1, 6>()[5][0] = 8;
    assert_eq!(array, [[1, 2, 3], [7, 5, 8]]);

    let mut flat = [1_u32, 2, 3, 4];
    assert_eq!(flat.unflatten::<2, 2>(), [[1, 2], [3, 4]]);
    assert_eq!(flat.unflatten_ref::<1, 4>(), &[[1], [2], [3], [4]]);
    flat.unflatten_mut::<4, 1>()[0][2] = 5;
    assert_eq!(flat, [1, 2, 5, 4]);

    // Output length is inferred from the context
    let inferred: &[u16; 6] = array.flatten_ref();
    assert_eq!(inferred, &[1, 2, 3, 7, 5, 8]);
}

#[test]
fn each() {
    let mut bytes = [[1_u8, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]];

    let pairs: &[[[u8; 2]; 2]] = array_reshape::unflatten_each_ref(&bytes);
    assert_eq!(
        pairs,
        &[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]
    );
    assert_eq!(array_reshape::flatten_each_ref::<_, 2, 2, 4>(pairs), &bytes);
    assert_eq!(
        array_reshape::rechunk_each_ref::<_, 2, 2, 1, 4>(pairs),
        &[
            [[1], [2], [3], [4]],
            [[5], [6], [7], [8]],
            [[9], [10], [11], [12]]
        ]
    );

    bytes.unflatten_each_mut::<2, 2>()[1][0][1] = 13;
    assert_eq!(bytes[1], [5, 13, 7, 8]);

    let mut chunks = [[[1_u16; 2]; 3]; 2];
    chunks.flatten_each_mut::<6>()[1][5] = 2;
    assert_eq!(chunks[1][2], [1, 2]);
    chunks.rechunk_each_mut::<3, 2>()[0][1][2] = 3;
    assert_eq!(chunks[0][2], [1, 3]);
    assert_eq!(chunks.flatten_each_ref::<6>()[1], [1, 1, 1, 1, 1, 2]);
    assert_eq!(chunks.rechunk_each_ref::<6, 1>()[0], [[1, 1, 1, 1, 1, 3]]);

    let empty: &[[[u8; 2]; 2]] = <[[u8; 4]]>::unflatten_each_ref(&[]);
    assert_eq!(empty, [[[0_u8; 2]; 2]; 0]);
}
