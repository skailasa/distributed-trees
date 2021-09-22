use std::time::{SystemTime};

use itertools::Itertools;
use mpi::traits::*;

use tree::data::{random};
use tree::morton::{
    Keys,
    Point,
    encode_points,
};

use tree::tree::{
    distribute_leaves,
    parallel_morton_sort,
};


fn main() {

    // 0.i Encode test points
    let points = random(1000);
    let depth = std::env::var("DEPTH").unwrap().parse().unwrap();
    let x0 = Point(0.5, 0.5, 0.5);
    let r0 = 0.5;
    let keys = encode_points(&points, &depth, &x0, &r0);

    let tree : Keys = keys.clone()
                          .into_iter()
                          .unique()
                          .collect();

    // 0.ii Setup MPI
    let start = SystemTime::now();

    // 0. Setup MPI
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    let root_rank = 0;
    let root_process = world.process_at_rank(root_rank);
    let nprocs : u16 = std::env::var("NPROCS").unwrap().parse().unwrap();

    // 1. Distribute the leaves
    let leaves = distribute_leaves(
        tree,
        world,
        rank,
        root_rank,
        root_process,
        depth as u16,
        nprocs
    );

    // 2. Perform parallel Morton sort over leaves
    parallel_morton_sort(
        leaves,
        world,
        rank,
        nprocs,
    );
}