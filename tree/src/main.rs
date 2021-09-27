use std::time::{SystemTime};

use itertools::Itertools;
use mpi::traits::*;

use tree::data::{random};
use tree::morton::{
    Key,
    Keys,
    Point,
    // encode_points,
    find_finest_common_ancestor,
    // find_siblings,
    find_children,
    find_parent,
    find_siblings,
    find_ancestors,
    encode_point,
    encode_points,
};

use tree::tree::{
    distribute_leaves,
    parallel_morton_sort,
    complete_region,
    find_blocks,
};


fn main() {


    // let a = Key(0, 0, 0, 3);
    // let b = Key(7, 7, 7, 3);
    // let depth = 3;

    // println!("ab = [");
    // println!("np.array([{}, {}, {}, {}]),", a.0, a.1, a.2, a.3);
    // println!("np.array([{}, {}, {}, {}])", b.0, b.1, b.2, b.3);
    // println!("]");

    // let res = complete_region(&a, &b, &depth);
    // assert!(b > a);
    // println!("complete = [");
    // for node in &res {
    //     println!("np.array([{}, {}, {}, {}], dtype=np.int64),", node.0, node.1, node.2, node.3);
    // };
    // println!("]")


    // 0.i Encode test points
    let points = random(100000);
    let depth = std::env::var("DEPTH").unwrap().parse().unwrap();
    let x0 = Point(0.5, 0.5, 0.5);
    let r0 = 0.5;
    let keys = encode_points(&points, &depth, &depth, &x0, &r0);


    let mut tree : Keys = keys.clone()
                          .into_iter()
                          .unique()
                          .collect();

    tree.sort();
    // println!("tree: {:?}", tree);

    // 0.ii Setup MPI
    let start = SystemTime::now();

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
        nprocs
    );

    // 2. Perform parallel Morton sort over leaves
    let leaves = parallel_morton_sort(
        leaves,
        world,
        rank,
        nprocs,
    );

    // 3. Complete minimal tree on each process, and find blocks.
    find_blocks(
        rank,
        leaves,
        &depth
    )

    // 4. Complete minimal block-tree across processes

    // 5. Rebalance blocks based on load

    // 6. Send points for blocks to each process

    // 7. Split blocks into adaptive tree, and pass into Octree structure.

    // 8. Compute interaction lists for each Octree

    // 9? Balance

}