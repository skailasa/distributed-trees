use std::time::{SystemTime};

use itertools::Itertools;
use mpi::traits::*;

use tree::data::{random};
use tree::morton::{
    Key,
    Keys,
    Point,
    encode_points,
    nearest_common_ancestor,
    find_siblings,
    find_children,
};

use tree::tree::{
    distribute_leaves,
    parallel_morton_sort,
    complete_region,
};


fn main() {

    let a = Key(0b10);
    let b = Key(0b0001110010);

    let na = nearest_common_ancestor(&a, &b);
    let w = find_children(&na);

    // println!("nearest common {}", na);
    // for node in w {
    //     println!("{}", node);
    // }

    let res = complete_region(&a, &b);

    assert!(b > a);
    for node in res {
        let level = node.0 & 0b1111;
        let k = ((node.0 >> 4) << 15) | level;

        println!("0b{:b},", k);
    }

    // let keys: Keys = vec![
    //     0b110001,
    //     0b1010001,
    //     0b1110001,
    //     0b1,
    //     0b1100001,
    //     0b10001,
    //     0b1000001,
    //     0b100001,
    // ].iter().map(|n| {Key(*n)}).collect();

    // let na = nearest_common_ancestor(&a, &b);
    // let cna = find_children(&na);
    // for key in cna {
    //     println!("{}", key);
    //     println!("{}", a < key);
    //     println!("{}\n", key < b);
    // }

    // let root: Key = Key(0);
    // let a: Key = Key(0b0001);
    // let b: Key = Key(1110001);

    // for key in find_children(&a) {
    //     println!("a: {}, key: {}", a, key);
    //     println!("{}", a < b);
    //     println!("{}", a < key);
    //     println!("{}\n", key < b);
    // }


    // // 0.i Encode test points
    // let points = random(1000);
    // let depth = std::env::var("DEPTH").unwrap().parse().unwrap();
    // let x0 = Point(0.5, 0.5, 0.5);
    // let r0 = 0.5;
    // let keys = encode_points(&points, &depth, &x0, &r0);

    // let tree : Keys = keys.clone()
    //                       .into_iter()
    //                       .unique()
    //                       .collect();

    // // 0.ii Setup MPI
    // let start = SystemTime::now();

    // let universe = mpi::initialize().unwrap();
    // let world = universe.world();
    // let rank = world.rank();

    // let root_rank = 0;
    // let root_process = world.process_at_rank(root_rank);
    // let nprocs : u16 = std::env::var("NPROCS").unwrap().parse().unwrap();

    // // 1. Distribute the leaves
    // let leaves = distribute_leaves(
    //     tree,
    //     world,
    //     rank,
    //     root_rank,
    //     root_process,
    //     nprocs
    // );

    // // 2. Perform parallel Morton sort over leaves
    // parallel_morton_sort(
    //     leaves,
    //     world,
    //     rank,
    //     nprocs,
    // );

    // 3. Complete minimal tree on each process

    // 4. Complete minimal block-tree

    // 5. Rebalance blocks based on load

    // 6. Send points for blocks to each process

    // 7. Split blocks into adaptive tree, and pass into Octree structure.

    // 8. Compute interaction lists for each Octree

    // 9? Balance

}