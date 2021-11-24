use mpi::traits::*;
use mpi::collective::{SystemOperation};
use mpi::topology::{UserCommunicator, SystemCommunicator, Color};

use tree::data::random;

use tree::morton::{Key, Point};
use tree::tree::{send_recv_kway, all_to_all_kway_i32};

fn main() {

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    let root_rank = 0;
    let k: i32 = 2;

    // Each process receives K-1 messages
    let mut recvbuf = vec![0 as i32];
    let mut sendbuf = vec![rank as i32];

    let world = world.split_by_color(Color::with_value(0)).unwrap();

    // send_recv_kway(world, rank, k, &sendbuf[..], &mut recvbuf[..]);
    // sendbuf.append(&mut recvbuf);
    // let mut recvbuf = vec![0 as i32; sendbuf.len() as usize];

    // let new_comm = send_recv_kway(comm, rank, k, &sendbuf[..], &mut recvbuf[..]);

    // sendbuf.append(&mut recvbuf);

    let received = all_to_all_kway_i32(world, rank, k, sendbuf, recvbuf);
    println!("RANK: {:?}, RECEIVES {:?}", rank, received);
    // println!("RANK: {:?}, RECEIVES {:?}", rank, received);


}