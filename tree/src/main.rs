use itertools::Itertools;
use std::time::{SystemTime};

use mpi::traits::*;

use mpi::{
    topology::{Rank},
    datatype::Partition,
    Count,
};


use tree::data::{random};
use tree::morton::{
    Key,
    MAX_POINTS,
    Leaf,
    Keys,
    Point,
    encode_points,
    keys_to_leaves,
};

use tree::tree::{
    SENTINEL,
    Weight,
    parallel_morton_sort,
    find_seeds,
    linearise,
    complete_blocktree,
    find_block_weights,
    block_partition,
    balance_load,
    unique_leaves
};

use tree::sort::{sample_sort};


fn main() {

    // 0.i Experimental Parameters
    let nprocs : u16 = std::env::var("NPROCS").unwrap().parse().unwrap_or(2);
    let depth: u64 = std::env::var("DEPTH").unwrap().parse().unwrap_or(3);
    let npoints = std::env::var("NPOINTS").unwrap().parse().unwrap_or(1000);
    // let ncrit = std::env::var("NCRIT").unwrap().parse().unwrap_or(MAX_POINTS);

    // Start timer
    let start = SystemTime::now();

    // 0.ii Setup Experiment and Distribute Leaves.
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    let root_rank: Rank = 0;
    let root_process = world.process_at_rank(root_rank);

    // Maximum possible buffer size
    let bufsize: usize = f64::powf(8.0, depth as f64) as usize;

    // Buffer for receiving partner keys
    let mut received_leaves = vec![Leaf::default(); bufsize];

    // 1. Generate random test points on a given process.
    let points = random(npoints);
    let x0 = Point(0.5, 0.5, 0.5);
    let r0 = 0.5;
    let keys = encode_points(&points, &depth, &depth, &x0, &r0);
    let mut local_leaves = keys_to_leaves(keys, points);

    // // Filter local leaves
    local_leaves = local_leaves.iter()
                               .filter(|&k| k.key != Key::default())
                               .cloned()
                               .collect();


    // 2. Perform parallel Morton sort over leaves
    let mut local_leaves = sample_sort(
            depth,
            &local_leaves,
            nprocs,
            rank,
            world,
    );


    // 3. Remove duplicates at each processor and remove overlaps if there are any
    let local_leaves = unique_leaves(local_leaves);

    // let min = local_leaves.iter().min().unwrap().key;
    // let max = local_leaves.iter().max().unwrap().key;
    // println!("RANK {} NLEAVES {:?}", rank, local_leaves.len());
    // println!("RANK {} LOCAL LEAVES MIN {:?} MAX {:?} nleaves {:?}", rank, min, max, local_leaves.len());
    // let mut npoints = 0;
    // for leaf in &local_leaves {
    //     npoints += leaf.npoints();
    // }
    // println!("RANK {} NPOINTS {:?}", rank, npoints);

    // 3. Complete minimal tree on each process, and find seed octants.
    let mut seeds = find_seeds(
        &local_leaves,
        &depth
    );

    // // // Debugging code
    // // println!("RANK {} SEEDS {:?}", rank, seeds);
    // // assert!(Key(0, 4, 4, 1) < Key(4, 0, 4, 1));

    // // 4. Complete minimal block-tree across processes

    // let mut local_blocktree = complete_blocktree(
    //     &mut seeds,
    //     &depth,
    //     rank,
    //     nprocs,
    //     world,
    // );

    // let mut weights = find_block_weights(&local_leaves, &local_blocktree, &depth);

    // // Debugging code
    // // println!("RANK {} LOCAL_BLOCKTREE {:?} ", rank, local_blocktree.len());
    // // // println!("RANK {} LOCAL_TREE {:?} ", rank, local_leaves);
    // // println!("RANK {} weights {:?} ", rank, weights);
    // // // println!("RANK {} LOCAL_SEEDS{:?} ", rank, seeds);

    // // 5. Re-balance blocks based on load
    // block_partition(
    //     weights,
    //     &mut local_blocktree,
    //     nprocs,
    //     rank,
    //     size,
    //     world,
    // );

    // println!("rank {} final blocks {:?}", rank, local_blocktree);
    // 6. Send points for blocks to each process


    // 7. Split blocks into adaptive tree, and pass into Octree structure.

    // 8. Compute interaction lists for each Octree

    // 9? Balance

}