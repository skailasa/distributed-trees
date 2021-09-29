use std::time::{SystemTime};
use std::cmp::Ordering;

use itertools::Itertools;
use mpi::traits::*;

use mpi::{
    collective::SystemOperation,
    point_to_point as p2p,
    topology::{Rank}
};


use tree::data::{random};
use tree::morton::{
    MAX_POINTS,
    SENTINELf64,
    Leaf,
    Leaves,
    LeafPoints,
    Ke,
    Key,
    Keys,
    Point,
    encode_points,
};

use tree::tree::{
    SENTINEL,
    Weight,
    parallel_morton_sort,
    find_seeds,
    linearise,
    complete_blocktree,
    find_block_weights,
    partition_blocks
};



fn main() {

    // // 0 Setup Experiment and Distribute Particle Data.
    // let start = SystemTime::now();

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    let root_rank: Rank = 0;
    let root_process = world.process_at_rank(root_rank);

    let nprocs : u16 = std::env::var("NPROCS").unwrap().parse().unwrap();
    let depth: u64 = std::env::var("DEPTH").unwrap().parse().unwrap();
    let npoints: u64 = std::env::var("NPOINTS").unwrap().parse().unwrap();
    let x0 = Point(0.5, 0.5, 0.5);
    let r0 = 0.5;

    // Maximum possible buffer size
    let bufsize: usize = f64::powf(8.0, depth as f64) as usize;

    // Buffer for keys local to this process

    // 1. Generate random points on each process
    let points = random(npoints);
    let keys = encode_points(&points, &depth, &depth, &x0, &r0);
    let mut points_to_keys: Vec<_> = points.iter().zip(keys.iter()).collect();
    points_to_keys.sort_by(|a, b| a.1.cmp(b.1));

    // Buffer for receiving partner keys
    let empty_points = LeafPoints([Point(SENTINELf64, SENTINELf64, SENTINELf64); MAX_POINTS]);
    let mut received_leaves = vec![Leaf{key: Key(0, 0, 0, SENTINEL), points: empty_points}; bufsize];

    let mut curr = Key(SENTINEL, SENTINEL, SENTINEL, SENTINEL);
    let mut local_tree: Keys = Vec::new();
    let mut leaf_indices: Vec<usize> = Vec::new();

    for (i, &(_, &key)) in points_to_keys.iter().enumerate() {
        if curr != key {
            curr = key;
            local_tree.push(curr);
            leaf_indices.push(i)
        }
    }

    leaf_indices.push(points_to_keys.len());

    let mut local_leaves: Leaves = Vec::new();

    // Fill up buffer of local leaf keys
    for (i, k) in local_tree.iter().enumerate() {
        // local_leaves[i] = k.clone();
        let mut points = LeafPoints([Point(SENTINELf64, SENTINELf64, SENTINELf64); MAX_POINTS]);
        let lidx = leaf_indices[i];
        let ridx = leaf_indices[i+1];

        for idx in lidx..ridx {
            let jdx = ridx-1-idx;
            points.0[jdx] = points_to_keys[idx].0.clone();
        }

        let leaf = Leaf{
            key: Key(k.0, k.1, k.2, k.3),
            points: points
        };
        local_leaves.push(leaf);
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
    // local_leaves = local_leaves.iter()
    //                            .unique()
    //                            .cloned()
    //                            .collect();


    // let min = local_leaves.iter().min().unwrap();
    // let max = local_leaves.iter().max().unwrap();
    // println!("rank {}, local leaves min {:?}", rank, min);

    // let local_leaves = linearise(&local_leaves, &depth);

    println!("rank {}, local leaves max {:?} nleaves {}", rank, local_leaves.iter().max().unwrap().key, local_leaves.len());
    println!("rank {}, local leaves min {:?} nleaves {}", rank, local_leaves.iter().min().unwrap().key, local_leaves.len());

    // 3. Complete minimal tree on each process, and find seed octants.
    // let mut seeds = find_seeds(
    //     &local_leaves,
    //     &depth
    // );

    // // 4. Complete minimal block-tree across processes
    // let mut local_blocktree = complete_blocktree(
    //     &mut seeds,
    //     &depth,
    //     rank,
    //     nprocs,
    //     world,
    // );

    // let mut weights = find_block_weights(&local_leaves, &local_blocktree, &depth);

    // // 5. Re-balance blocks based on load
    // partition_blocks(
    //     weights,
    //     &mut local_blocktree,
    //     nprocs,
    //     rank,
    //     size,
    //     world,
    // );

    // println!("rank {} final blocks {:?}", rank, local_blocktree);
    // 6. Send points for blocks to each process

    // let mut points = [Point(0., 0., 0.); 500];
    // let rnd = random(1);
    // for (i, &point) in rnd.iter().enumerate() {
    //     points[i] = point.clone();
    // }

    // let test_points = LeafPoints(points);

    // let ke = Leaf(rank as u64, 0, 0, 0, test_points.clone());
    // let mut buf = Leaf(rank as u64, 0, 0, 0, test_points.clone());
    // let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
    // let next_process = world.process_at_rank(next_rank);

    // p2p::send_receive_into(&ke, &next_process, &mut buf, &next_process);

    // println!("rank {} partner rank {}", rank, next_rank);
    // println!("received at rank {} the msg {:?}", rank, buf);

    // 7. Split blocks into adaptive tree, and pass into Octree structure.

    // 8. Compute interaction lists for each Octree

    // 9? Balance
}