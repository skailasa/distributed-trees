use rand::{thread_rng, Rng};

extern crate mpi;

use::mpi::request;
use mpi::traits::{Source, Destination, MatchedReceiveVec};
use mpi::{topology::{Rank, SystemCommunicator}};
use mpi::datatype::{MutView, UserDatatype, View};
use mpi::traits::*;
use mpi::Count;

use crate::morton::{Leaf, Leaves, Key};

// Sample density
const K: usize = 10;


pub fn sample_sort(
    depth: u64,
    local_leaves: &Leaves,
    nprocs: u16,
    rank: Rank,
    world: SystemCommunicator
) -> Leaves {

    // Buffer for receiving partner keys
    let mut received_leaves: Leaves = Vec::new();

     let mut received_samples = vec![Leaf::default(); K*(nprocs as usize)];
    let nleaves = local_leaves.len();
    // 1. Collect 'k' samples from each process that isn't the root.
    let mut rng = thread_rng();
    let mut sample_idxs: Vec<usize> = (0..K).map(|_| rng.gen_range(0..nleaves)).collect();

    let mut local_samples: Leaves = vec![Leaf::default(); K];

    for (i, &sample_idx) in sample_idxs.iter().enumerate() {
        local_samples[i] = local_leaves[sample_idx].clone();
    }

    // println!("rank {} sample_idxs {:?}", rank, sample_idxs );
    // for s in &local_samples {
    //     println!("rank {} local samples {:?}", rank, s.key);
    // }

    world.all_gather_into(&local_samples[..], &mut received_samples[..]);

    // Ignore first k samples to ensure (nproc-1) splitters
    received_samples.sort();
    received_samples = received_samples[K..].to_vec();


    // Every k'th sample defines a bucket.
    let mut splitters: Leaves = received_samples.iter()
                                                .step_by(K)
                                                .cloned()
                                                .collect();

    let nsplitters = splitters.len();

    let max = received_samples.iter().max().unwrap().clone();

    // 2. Sort local leaves into buckets

    let min = received_samples.iter().min().unwrap().clone().key;
    // println!("rank {} local leaves {:?} {:?}", rank, min, max.key);
    // println!("rank {} splitter {:?} {}", rank, splitters.iter().min().unwrap().key, splitters.len());

    let mut buckets: Vec<Leaves> = vec![Vec::new(); nprocs as usize];

    for leaf in local_leaves.iter() {

        for i in 0..(nprocs as usize) {
            if i < nsplitters {
                let s = &splitters[i];
                if leaf < s  {
                    buckets[i].push(leaf.clone());
                    break;
            }
            } else {
                buckets[i].push(leaf.clone())
            }
        }
    }

    // 3. Send all local buckets to their matching processor.

    for i in 0..(nprocs as i32) {

        if rank != i {
            let msg = &buckets[i as usize];
            world.process_at_rank(i).send(&msg[..]);
        } else {
            for _ in 1..world.size() {
                let (mut msg, status) = world.any_process().receive_vec::<Leaf>();
                // println!(
                //     "Process {} got long message {:?}.\nStatus is: {:?}",
                //     rank, msg.len(), status
                // );

                received_leaves.append(&mut msg);
            }
        }
        world.barrier();
    }



    // 4. Sort leaves on matching processors.
    received_leaves.sort();

    received_leaves
}



