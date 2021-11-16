use std::time::Instant;

use mpi::traits::*;
use mpi::collective::{SystemOperation};

use tree::data::random;

use tree::morton::{Key, Point};
use tree::tree::unbalanced_tree;

fn main() {

    // Setup MPI
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    let root_rank = 0;

    // 0. Experimental Parameters
    let depth: u64 = std::env::var("DEPTH").unwrap().parse().unwrap_or(3);
    // let npoints: u64 = std::env::var("NPOINTS").unwrap().parse().unwrap_or(1000);
    let ncrit: usize = std::env::var("NCRIT").unwrap().parse().unwrap_or(1000);
    let n_max: u64 = 32;
    let n: u64 = n_max/(size as u64);
    let npoints: u64 = n*(1000000);

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

    // Generate distributed unbalanced tree from a set of distributed points
    let (unbalanced, times) = unbalanced_tree(&depth, &ncrit, &universe, &mut points, x0, r0);

    world.barrier();

    // broadcast total number of leaves into root rank
    let nleaves = unbalanced.len() as u32;
    let mut sum = 0;

    // Print runtime to stdout
    if rank == root_rank {
        world
            .process_at_rank(root_rank)
            .reduce_into_root(&nleaves, &mut sum, SystemOperation::sum());
        /// universe size, number of leaves, total runtime, encoding time, sorting time
        println!(
            "{:?}, {:?}, {:?}, {:?}, {:?}",
            size,
            sum,
            times.get(&"total".to_string()),
            times.get(&"encoding".to_string()),
            times.get(&"sorting".to_string())
        )

    } else {
        world
            .process_at_rank(root_rank)
            .reduce_into(&nleaves, SystemOperation::sum())
    }
}
