use std::time::Instant;

use mpi::traits::*;
use mpi::collective::{SystemOperation};
use mpi::topology::{UserCommunicator, SystemCommunicator, Color};

use tree::data::random;

use tree::morton::{Key, Point};
use tree::tree::{Times, send_recv_kway, all_to_all_kway_i32, all_to_all_kwayv_i32, all_to_all};

fn main() {

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let world = world.split_by_color(Color::with_value(0)).unwrap();
    let size = world.size();
    let rank = world.rank();
    let root_rank = 0;
    let k: i32 = 4;

    // Each process receives K-1 messages
    let msgs = vec![rank; ((rank+1)*(rank+1)) as usize];
    let buckets = Vec::new();

    for i in 1..size {
        buckets.insert(&msgs.clone());
    }

    let world = universe.world();
    let world = world.split_by_color(Color::with_value(0)).unwrap();

    let times: Times = HashMap::new();


    let kway = Instant::now();
    all_to_all_kwayv_i32(world, rank, k, msgs);
    time.insert("encoding".to_string(), kway.elapsed().as_millis());

    let intrinsic = Instant::now();
    all_to_all(world, size, buckets);
    time.insert("intrinsic".to_string(), intrinsic.elapsed().as_millis());

    if rank == root_rank {
        println!(
            "{:?}, {:?}, {:?}, {:?}, {:?}",
            size,
            sum,
            times.get(&"total".to_string()).unwrap(),
            times.get(&"kway".to_string()).unwrap(),
            times.get(&"intrinsic".to_string()).unwrap()
        )
    }
}