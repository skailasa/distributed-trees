use std::time::Instant;

use mpi::traits::*;

use tree::data::random;

use tree::morton::{Key, Point};
use tree::tree::unbalanced_tree;

fn main() {
    // 0. Experimental Parameters
    let depth: u64 = std::env::var("DEPTH").unwrap().parse().unwrap_or(3);
    let npoints: u64 = std::env::var("NPOINTS").unwrap().parse().unwrap_or(1000);
    let ncrit: usize = std::env::var("NCRIT").unwrap().parse().unwrap_or(1000);

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
    let universe = mpi::initialize().unwrap();

    // Start timer
    let start = Instant::now();

    // 1. Generate distributed unbalanced tree from a set of distributed points
    let unbalanced = unbalanced_tree(&depth, &ncrit, universe, &mut points, x0, r0);

    // 2. Balance the distributed tree

    // 3. Perform load balance based on interaction list density.

    // 4. Form locally essential Octree

    // Print runtime to stdout
    println!("RUNTIME: {:?} ms", start.elapsed().as_millis());
}
