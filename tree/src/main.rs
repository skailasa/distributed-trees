use std::time::{SystemTime};

use itertools::Itertools;
use mpi::traits::*;

use mpi::{
    datatype::{Partition},
};


use tree::data::{random};
use tree::morton::{
    Key,
    Keys,
    Point,
    encode_points,
};

use tree::tree::{
    SENTINEL,
    balance_load,
    parallel_morton_sort,
    find_seeds,
    linearise,
    find_deepest_first_descendent,
    find_deepest_last_descendent,
};




fn main() {

    // 0.i Setup Experiment and Distribute Leaves.
    let start = SystemTime::now();

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    let root_rank = 0;
    let root_process = world.process_at_rank(root_rank);

    let nprocs : u16 = std::env::var("NPROCS").unwrap().parse().unwrap();
    let depth: u64 = std::env::var("DEPTH").unwrap().parse().unwrap();
    let npoints = std::env::var("NPOINTS").unwrap().parse().unwrap();

    // Maximum possible buffer size
    let bufsize: usize = (f64::powf(8.0, depth as f64) / (nprocs as f64)).ceil() as usize;

    // Buffer for keys local to this process
    let mut local_leaves = vec![Key(0, 0, 0, SENTINEL); bufsize];

    // Buffer for receiving partner keys
    let mut received_leaves = vec![Key(0, 0, 0, SENTINEL); bufsize];


    // 1. Generate random points on each process.
    let points = random(npoints);
    let x0 = Point(0.5, 0.5, 0.5);
    let r0 = 0.5;
    let keys = encode_points(&points, &depth, &depth, &x0, &r0);


    let local_tree: Keys = keys.iter()
                               .unique()
                               .cloned()
                               .collect();

    // println!("RANK {} nleaves in total {}", rank, local_tree.len());
    for (i, k) in local_tree.iter().enumerate() {
        local_leaves[i] = k.clone();
    }

    // 2. Perform parallel Morton sort over leaves
    let mut local_leaves = parallel_morton_sort(
        local_leaves,
        received_leaves,
        world,
        rank,
        nprocs,
    );

    // 3. Remove duplicates at each processor and remove overlaps if there are any
    local_leaves = local_leaves.iter()
                               .unique()
                               .cloned()
                               .collect();

    let mut local_leaves = linearise(&local_leaves, &depth);

    // // Debugging code
    // let local_min = local_leaves.iter().min().unwrap().clone();
    // let local_max = local_leaves.iter().max().unwrap().clone();
    // for &leaf in &local_leaves {
    //     if (leaf != local_max) & (leaf != local_min) {
    //         assert!((local_min < leaf) & (leaf < local_max));
    //     }
    // }

    // assert!(local_min < local_max);

    // println!("RANK {} min {:?} max {:?}", rank, local_min, local_max);
    // println!("RANK {} nleaves {:?}", rank, local_leaves.len());


    // 3. Complete minimal tree on each process, and find seed octants.
    let seeds = find_seeds(
        rank,
        local_leaves,
        &depth
    );

    // // Debugging code
    // println!("RANK {} SEEDS {:?}", rank, seeds);
    // assert!(Key(0, 4, 4, 1) < Key(4, 0, 4, 1));

    // 4. Complete minimal block-tree across processes
    let root = Key(0, 0, 0, 0);
    let dld_root = find_deepest_last_descendent(&root, &depth);
    let dfd_root = find_deepest_first_descendent(&root, &depth);
    println!("DLD ROOT {:?} DEPTH {}", dld_root, depth);
    println!("DFD ROOT {:?} DEPTH {}", dfd_root, depth);

    // 5. Rebalance blocks based on load

    // 6. Send points for blocks to each process

    // 7. Split blocks into adaptive tree, and pass into Octree structure.

    // 8. Compute interaction lists for each Octree

    // 9? Balance

}