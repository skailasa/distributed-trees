use mpi::traits::*;
use mpi::collective::{SystemOperation};
use mpi::topology::{UserCommunicator, SystemCommunicator, Color};

use tree::data::random;

use tree::morton::{Key, Point};
use tree::tree::{send_recv_kway, all_to_all_kway_i32, all_to_all_kwayv_i32};

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

    let world = universe.world();
    let world = world.split_by_color(Color::with_value(0)).unwrap();

    let rec = all_to_all_kwayv_i32(world, rank, k, msgs);

    println!("RANK: {:?} FINAL {:?}", rank, rec);

}