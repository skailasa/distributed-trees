use std::time::{Instant};

use mpi::traits::*;

use tree::data::random;
use tree::morton::{encode_points, keys_to_leaves, Key, Point};

use tree::tree::{
    assign_blocks_to_leaves, block_partition, complete_blocktree, find_block_weights, find_seeds,
    transfer_leaves_to_coarse_blocktree, transfer_leaves_to_final_blocktree, unique_leaves,
};

use tree::sort::sample_sort;

fn main() {
    // 0.i Experimental Parameters
    let nprocs: u16 = std::env::var("NPROCS").unwrap().parse().unwrap_or(2);
    let depth: u64 = std::env::var("DEPTH").unwrap().parse().unwrap_or(3);
    let npoints = std::env::var("NPOINTS").unwrap().parse().unwrap_or(1000);

    // 0.ii Setup Experiment and Distribute Leaves.
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    let start = Instant::now();

    // 1. Generate random test points on a given process.
    let points = random(npoints);
    let x0 = Point(0.5, 0.5, 0.5);
    let r0 = 0.5;
    let keys = encode_points(&points, &depth, &depth, &x0, &r0);
    let mut local_leaves = keys_to_leaves(keys, points, true);

    // Filter local leaves
    local_leaves = local_leaves
        .iter()
        .filter(|&k| k.key != Key::default())
        .cloned()
        .collect();

    // 2. Perform parallel Morton sort over leaves
    let local_leaves = sample_sort(&local_leaves, nprocs, rank, world);

    // 3. Remove duplicates at each processor and remove overlaps if there are any
    let mut local_leaves = unique_leaves(local_leaves);

    // 4.i Complete minimal tree on each process, and find seed octants.
    let mut seeds = find_seeds(&local_leaves, &depth);

    // 4.ii If leaf is less than the minimum seed in a given process,
    // it needs to be sent to the previous process
    local_leaves.sort();

    let mut local_leaves =
        transfer_leaves_to_coarse_blocktree(&local_leaves, &seeds, rank, world, nprocs as i32);

    // 5. Complete minimal block-tree across processes
    let mut local_blocktree = complete_blocktree(&mut seeds, &depth, rank, nprocs, world);

    // Associate leaves with blocks
    assign_blocks_to_leaves(&mut local_leaves, &local_blocktree, &depth);

    let weights = find_block_weights(&local_leaves, &local_blocktree);

    // 6.i Re-balance blocks based on load, find out which blocks were sent to partner.
    let sent_blocks = block_partition(weights, &mut local_blocktree, nprocs, rank, size, world);

    // 6.ii For each sent block, send corresponding leaves to partner process.
    let local_leaves =
        transfer_leaves_to_final_blocktree(&sent_blocks, local_leaves, nprocs, rank, world);

    println!("RANK {} final {}", rank, local_leaves.len());

    // 7. Split blocks into adaptive tree, and pass into Octree structure.

    // 8. Compute interaction lists for each Octree

    // 9? Balance

    // Print runtime to stdout
    world.barrier();
    println!("RANK {} RUNTIME: {:?}", rank, start.elapsed().as_secs());
}
