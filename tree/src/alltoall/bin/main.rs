use std::time::Instant;
use std::collections::HashMap;

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
    // let mut msgs = vec![rank; 1 as usize];
    // println!("RANK {:?} msgs {:?}", rank, msgs);
    // let mut recvbuf = vec![0; (size) as usize];
    // let received = all_to_all_kway_i32(world, rank, k, msgs);
    // println!("RANK {:?} received {:?}", rank, received);

    let mut msgs = vec![rank; (rank+1) as usize];
    println!("msgs {:?}", msgs);



    // let mut buckets: Vec<Vec<i32>> = vec![Vec::new(); (size-1) as usize];

    // for i in 0..(size-1) {
    //     for msg in &msgs {
    //         buckets[i as usize].push(msg.clone());
    //     }
    // }

    // let mut times: Times = HashMap::new();


    // let kway = Instant::now();
    // let a = all_to_all_kwayv_i32(world, rank, k, msgs);
    // times.insert("kway".to_string(), kway.elapsed().as_millis());

    // let world = universe.world();
    // let intrinsic = Instant::now();
    // let b = all_to_all(world, size, buckets);
    // times.insert("intrinsic".to_string(), intrinsic.elapsed().as_millis());

    // if rank == root_rank {
    //     println!(
    //         "{:?}, {:?}, {:?}",
    //         size,
    //         times.get(&"kway".to_string()).unwrap(),
    //         times.get(&"intrinsic".to_string()).unwrap()
    //     );
    // }


}