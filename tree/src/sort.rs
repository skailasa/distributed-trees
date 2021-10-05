use rand::{thread_rng, Rng};

extern crate mpi;

use mpi::topology::{Rank, SystemCommunicator};
use mpi::traits::*;
use mpi::traits::{Destination, Source};

use crate::morton::{Leaf, Leaves};

// Sample density
const K: usize = 10;

pub fn sample_sort(
    local_leaves: &Leaves,
    nprocs: u16,
    rank: Rank,
    world: SystemCommunicator,
) -> Leaves {
    // Buffer for receiving partner keys
    let mut received_leaves: Leaves = Vec::new();

    let mut received_samples = vec![Leaf::default(); K * (nprocs as usize)];
    let nleaves = local_leaves.len();

    // 1. Collect 'K' samples from each process onto all other processes
    let mut rng = thread_rng();
    let sample_idxs: Vec<usize> = (0..K).map(|_| rng.gen_range(0..nleaves)).collect();

    let mut local_samples: Leaves = vec![Leaf::default(); K];

    for (i, &sample_idx) in sample_idxs.iter().enumerate() {
        local_samples[i] = local_leaves[sample_idx].clone();
    }

    world.all_gather_into(&local_samples[..], &mut received_samples[..]);

    // Ignore first K samples to ensure (nproc-1) splitters
    received_samples.sort();
    received_samples = received_samples[K..].to_vec();

    // Every K'th sample defines a bucket.
    let splitters: Leaves = received_samples.iter().step_by(K).cloned().collect();
    let nsplitters = splitters.len();

    // 2. Sort local leaves into buckets
    let mut buckets: Vec<Leaves> = vec![Vec::new(); nprocs as usize];

    for leaf in local_leaves.iter() {
        for i in 0..(nprocs as usize) {
            if i < nsplitters {
                let s = &splitters[i];
                if leaf < s {
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
                let (mut msg, _) = world.any_process().receive_vec::<Leaf>();
                received_leaves.append(&mut msg);
            }
        }
        world.barrier();
    }

    // 4. Sort leaves on matching processors.
    received_leaves.append(&mut buckets[rank as usize]);
    received_leaves.sort();
    received_leaves
}
