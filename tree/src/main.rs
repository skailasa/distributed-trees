use std::time::Instant;

use mpi::traits::*;

use tree::data::random;

use tree::morton::{Point};
use tree::tree::{unbalanced_tree};

fn main() {
    // 0.i Experimental Parameters
    let depth: u64 = std::env::var("DEPTH").unwrap().parse().unwrap_or(3);
    let npoints: u64 = std::env::var("NPOINTS").unwrap().parse().unwrap_or(1000);
    let ncrit : u64 = std::env::var("NCRIT").unwrap().parse().unwrap_or(1000);


    // 0.ii Setup Experiment and Distribute Leaves.

    let start = Instant::now();

    // 1. Generate random test points on a given process.
    let points = random(npoints);
    let x0 = Point(0.5, 0.5, 0.5);
    let r0 = 0.5;
    let universe = mpi::initialize().unwrap();

    let unbalanced = unbalanced_tree(&depth, &ncrit, universe, points, x0, r0);


    // Print runtime to stdout
    println!("RUNTIME: {:?} ms", start.elapsed().as_millis());
}
