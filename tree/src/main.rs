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
    find_blocks,
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
    let mut received_leaves = vec![Key(0, 0, 0, SENTINEL); bufsize];

    // Buffer for receiving partner keys

    if rank == root_rank {

        let points = random(npoints);
        let x0 = Point(0.5, 0.5, 0.5);
        let r0 = 0.5;
        let keys = encode_points(&points, &depth, &depth, &x0, &r0);


        let mut tree : Keys = keys.clone()
                            .into_iter()
                            .unique()
                            .collect();

        println!("nleaves in total {}", tree.len());

        // Calculate load
        let load = balance_load(tree.len() as u16, nprocs);
        let partition = Partition::new(&tree[..], load.counts, load.displs);

        // ScatterV
        root_process.scatter_varcount_into_root(&partition, &mut local_leaves[..])

    } else {
        root_process.scatter_varcount_into(&mut local_leaves[..])
    }

    world.barrier();

    // 2. Perform parallel Morton sort over leaves
    let mut local_leaves = parallel_morton_sort(
        local_leaves,
        received_leaves,
        world,
        rank,
        nprocs,
    );


    let local_min = local_leaves.iter().min().unwrap().clone();
    let local_max = local_leaves.iter().max().unwrap().clone();
    for &leaf in &local_leaves {
        if (leaf != local_max) & (leaf != local_min) {
            assert!((local_min < leaf) & (leaf < local_max));
        }
    }

    assert!(local_min < local_max);
    assert!(Key(4, 0, 0, 1) > Key(0, 4, 0, 1));

    println!("RANK {} min {:?} max {:?}", rank, local_min, local_max);
    println!("RANK {} nleaves {:?}", rank, local_leaves.len());
    // 3. Complete minimal tree on each process, and find blocks.
    let blocks = find_blocks(
        rank,
        local_leaves,
        &depth
    );

    println!("RANK {} BLOCKS {:?}", rank, blocks);

    // 4. Complete minimal block-tree across processes

    // 5. Rebalance blocks based on load

    // 6. Send points for blocks to each process

    // 7. Split blocks into adaptive tree, and pass into Octree structure.

    // 8. Compute interaction lists for each Octree

    // 9? Balance

}