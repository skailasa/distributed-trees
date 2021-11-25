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
    let k: i32 = 2;

    // Each process receives K-1 messages
    let mut recvbuf = vec![0 as i32; (k-1) as usize];
    let msgs = vec![rank; ((rank+1)*(rank+1)) as usize];
    let mut sendbuf = vec![msgs.len() as i32];


    // send_recv_kway(world, rank, k, &sendbuf[..], &mut recvbuf[..]);
    // sendbuf.append(&mut recvbuf);
    // let mut recvbuf = vec![0 as i32; sendbuf.len() as usize];

    // let new_comm = send_recv_kway(comm, rank, k, &sendbuf[..], &mut recvbuf[..]);

    // sendbuf.append(&mut recvbuf);

    let mut recv_msg_sizes = vec![0 as i32; (k-1) as usize];
    let mut send_msg_sizes = vec![msgs.len() as i32];

    // Find size of all messages received by this rank.
    let msg_sizes = all_to_all_kway_i32(world, rank, k, send_msg_sizes, recv_msg_sizes);
    let msg_sizes: Vec<usize> = msg_sizes.into_iter().map(|x| x as usize).collect();
    println!("RANK: {:?}, RECEIVES {:?}", rank, msg_sizes);

    // Use this sizes to send messages of variable length

    let world = universe.world();
    let world = world.split_by_color(Color::with_value(0)).unwrap();

    let rec = all_to_all_kwayv_i32(world, rank, k, msgs, msg_sizes);

    println!("RANK: {:?} FINAL {:?}", rank, rec);

}