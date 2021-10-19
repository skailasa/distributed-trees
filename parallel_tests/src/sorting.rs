extern crate mpi;
extern crate tree;

use mpi::environment::Universe;
use mpi::traits::*;

use tree::data::random;
use tree::morton::{encode_points, Key, Leaves, Point, Points};
use tree::tree::sample_sort;


// Test sample sort
pub fn test_sample_sort(universe: Universe) {
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    let depth: u64 = 3;
    let npoints: u64 = 10000;
    let ncrit: usize = 150;

    // Generate random test points on a given process.
    let mut points = random(npoints);
    let x0 = Point {
        x: 0.5,
        y: 0.5,
        z: 0.5,
        global_idx: 0,
        key: Key::default(),
    };
    let r0 = 0.5;

    if rank == 0 {
        println!(
            "Test Sample Sort with {} points across {} processes",
            npoints, size
        );
    }

    // 1. Encode points to leaf keys inplace.
    encode_points(&mut points, &depth, &depth, &x0, &r0);

    // Temporary buffer for receiving partner keys
    let mut sorted_leaves: Leaves = Vec::new();
    let mut sorted_points: Points = Vec::new();

    // 2. Perform parallel Morton sort over points
    sample_sort(
        &mut points,
        &ncrit,
        &mut sorted_leaves,
        &mut sorted_points,
        size,
        rank,
        world,
    );

    // Test that the maximum on this process is less than the minimum on the next process
    let prev_rank = if rank > 0 { rank - 1 } else { size - 1 };
    if rank > 0 {
        let min: Key = sorted_leaves.iter().min_by_key(|p| p.key).unwrap().key as Key;
        world.process_at_rank(prev_rank).send(&min);
    }
    if rank < (size - 1) {
        let (rec, _) = world.any_process().receive_vec::<Key>();
        let max: Key = sorted_leaves.iter().max_by_key(|p| p.key).unwrap().key as Key;
        assert!(max <= rec[0]);
    }

    // Test that leaves are sorted on this process
    let mut prev = sorted_leaves[0];
    for &leaf in sorted_leaves.iter().skip(1) {
        assert!(leaf >= prev);
        prev = leaf;
    }
}
