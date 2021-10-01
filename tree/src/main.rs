use std::time::{SystemTime};

use mpi::traits::*;

use tree::data::{random};
use tree::morton::{
    Key,
    Point,
    encode_points,
    keys_to_leaves,
};

use tree::tree::{
    find_seeds,
    complete_blocktree,
    find_block_weights,
    block_partition,
    unique_leaves
};

use tree::sort::{sample_sort};


fn main() {

    // 0.i Experimental Parameters
    let nprocs : u16 = std::env::var("NPROCS").unwrap().parse().unwrap_or(2);
    let depth: u64 = std::env::var("DEPTH").unwrap().parse().unwrap_or(3);
    let npoints = std::env::var("NPOINTS").unwrap().parse().unwrap_or(1000);


    // 0.ii Setup Experiment and Distribute Leaves.
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    // Start timer on root process
    if rank == 0 {
        let start = SystemTime::now();
    }

    // 1. Generate random test points on a given process.
    let points = random(npoints);
    let x0 = Point(0.5, 0.5, 0.5);
    let r0 = 0.5;
    let keys = encode_points(&points, &depth, &depth, &x0, &r0);
    let mut local_leaves = keys_to_leaves(keys, points);

    // Filter local leaves
    local_leaves = local_leaves.iter()
                               .filter(|&k| k.key != Key::default())
                               .cloned()
                               .collect();


    // 2. Perform parallel Morton sort over leaves
    let local_leaves = sample_sort(
            &local_leaves,
            nprocs,
            rank,
            world,
    );

    // 3. Remove duplicates at each processor and remove overlaps if there are any
    let local_leaves = unique_leaves(local_leaves);

    // 4. Complete minimal tree on each process, and find seed octants.
    let mut seeds = find_seeds(
        &local_leaves,
        &depth
    );

    // 5 Complete minimal block-tree across processes
    let mut local_blocktree = complete_blocktree(
        &mut seeds,
        &depth,
        rank,
        nprocs,
        world,
    );

    let weights = find_block_weights(&local_leaves, &local_blocktree, &depth);

    // Associate leaves with blocks
    // ...

    // 6. Re-balance blocks based on load
    // block_partition(
    //     weights,
    //     &mut local_blocktree,
    //     nprocs,
    //     rank,
    //     size,
    //     world,
    // );

    println!("blocks_{} = np.array([", rank);
    for s in local_blocktree {
        println!("[{}, {}, {}, {}],", s.0, s.1, s.2, s.3);
    }
    println!("])");

    // 6. Send leafs to their associated process via their block.

    // 7. Split blocks into adaptive tree, and pass into Octree structure.

    // 8. Compute interaction lists for each Octree

    // 9? Balance

    // Print runtime to stdout
    if rank == 0 {
        let runtime = SystemTime::now().elapsed().unwrap();
        println!("RUNTIME: {:?}", runtime);
    }
}