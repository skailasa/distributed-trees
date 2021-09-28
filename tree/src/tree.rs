use std::collections::HashSet;

use memoffset::offset_of;
use mpi::{
    Address,
    datatype::{Equivalence, UserDatatype, UncommittedUserDatatype},
    Count,
    point_to_point as p2p,
    traits::*,
    topology::{SystemCommunicator},
    collective::SystemOperation,
    topology::{Rank}
};

use crate::morton::{
    Key,
    Keys,
    find_ancestors,
    find_descendents,
    find_children,
    find_finest_common_ancestor,
    find_deepest_first_descendent,
    find_deepest_last_descendent
};

pub const SENTINEL: u64 = 999;
pub const MPI_PROC_NULL: i32 = -1;

/// Implement MPI equivalent datatype for Morton keys
unsafe impl Equivalence for Key {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1, 1, 1, 1],
            &[
                offset_of!(Key, 0) as Address,
                offset_of!(Key, 1) as Address,
                offset_of!(Key, 2) as Address,
                offset_of!(Key, 3) as Address,
            ],
            &[
                UncommittedUserDatatype::contiguous(1, &u64::equivalent_datatype()).as_ref(),
                UncommittedUserDatatype::contiguous(1, &u64::equivalent_datatype()).as_ref(),
                UncommittedUserDatatype::contiguous(1, &u64::equivalent_datatype()).as_ref(),
                UncommittedUserDatatype::contiguous(1, &u64::equivalent_datatype()).as_ref(),
            ],
        )
    }
}


#[derive(Debug, Copy, Clone)]
pub struct Weight(pub u64);
pub type Weights = Vec<Weight>;

unsafe impl Equivalence for Weight {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1],
            &[offset_of!(Weight, 0) as Address],
            &[UncommittedUserDatatype::contiguous(1, &u64::equivalent_datatype()).as_ref()]
        )
    }
}


/// Merge two sorted vectors of Morton keys
///
/// # Arguments
/// `a` - Sorted vector of Morton keys
/// `b` - Sorted vector of Morton keys
fn merge(a: &Keys, b: &Keys) -> Vec<Key> {

    let mut merged = Vec::new();

    let mut pt_a : usize = 0;
    let mut pt_b : usize = 0;

    while pt_a < a.len() && pt_b < b.len() {
        if a[pt_a] < b[pt_b] {
            merged.push(a[pt_a].clone());
            pt_a += 1;
        } else {
            merged.push(b[pt_b].clone());
            pt_b += 1;
        }
    }

    merged.append(&mut a[pt_a..].to_vec());
    merged.append(&mut b[pt_b..].to_vec());

    merged
}


/// Calculate the middle element of a vector
fn middle(nelems: usize) -> usize {

    if nelems % 2 == 0 {
        (nelems / 2) as usize
    } else {
        ((nelems - 1)/2) as usize
    }
}

/// MPI Load calculations, and memory displacements for distributing octree
#[derive(Debug)]
pub struct Load {
    pub counts : Vec<Count>,
    pub displs : Vec<Count>,
}

/// Non optimal load balance of a given number of tasks using a given
/// number of of processes.
///
/// # Arguments
/// * `ntasks` - Number of tasks
/// * `nprocs` - Number of processors
pub fn balance_load(ntasks: u16, nprocs: u16) -> Load {

    let tasks_per_process : f32 = ntasks as f32 / nprocs as f32;
    let tasks_per_process : i32 = tasks_per_process.ceil() as i32;
    let remainder : i32 = (ntasks as i32) - (tasks_per_process*((nprocs-1) as i32));

    let mut counts: Vec<Count> = vec![tasks_per_process; nprocs as usize];
    counts[(nprocs-1) as usize] = remainder;

    let displs: Vec<Count> = counts
        .iter()
        .scan(0, |acc, &x| {
            let tmp = *acc;
            *acc += x;
            Some(tmp)
        })
        .collect();

    Load {
        counts,
        displs
    }
}


/// Compute rank of the partner process in odd/even sort algorithm
/// for the given process.
///
/// # Arguments
/// `rank` - MPI rank of the given process
/// `phase` - Phase of the odd/even sort algorithm
/// `nprocs` - Size of the MPI communicator being used
pub fn compute_partner(rank: i32, phase: i32, nprocs: i32) -> i32 {

    let mut partner;

    if (phase % 2) == 0 {
        if rank % 2 != 0 {
            partner = rank - 1;
        } else {
            partner = rank + 1;
        }
    } else {
        if rank % 2 != 0 {
            partner = rank + 1;
        } else {
            partner = rank - 1;
        }
    };

    if partner == -1 || partner == nprocs  {
        partner = MPI_PROC_NULL;
    }
    partner
}


/// Perform parallel sort on leaves, dominates complexity of algorithm
pub fn parallel_morton_sort(
    mut local_leaves: Keys,
    mut received_leaves: Keys,
    world: SystemCommunicator,
    rank: i32,
    nprocs: u16,
) -> Keys {
    // Guaranteed to converge in nprocs phases
    for phase in 0..nprocs {

        let partner = compute_partner(rank, phase as i32, nprocs as i32);

        // Send local leaves to partner, and receive their leaves
        if partner != MPI_PROC_NULL {

            let partner_process = world.process_at_rank(partner);

            p2p::send_receive_into(&local_leaves[..], &partner_process, &mut received_leaves[..], &partner_process);

            // println!("Rank {}, received: {:?}  sent {:?}", rank, received_leaves, local_leaves);

            // Perform merge, excluding sentinel from received leaves
            let mut received: Keys = received_leaves.iter()
                                                .filter(|&k| k.3 != SENTINEL)
                                                .cloned()
                                                .collect();

            let mut local: Keys = local_leaves.iter()
                                          .filter(|&k| k.3 != SENTINEL)
                                          .cloned()
                                          .collect();

            // Input to merge must be sorted
            local.sort();
            received.sort();
            let merged = merge(&local, &received);

            let mid = middle(merged.len());

            if rank < partner {
                // Keep smaller keys
                local_leaves = merged[..mid].to_vec();

            } else {
                // Keep larger keys
                local_leaves = merged[mid..].to_vec();
            }
        }
    }

    local_leaves.sort();
    local_leaves
}


/// Construct a minimal linear octree between two octants adapted from
/// Algorithm 3 in Sundar et. al.
pub fn complete_region(a: &Key, b: &Key, depth: &u64) -> Keys {

    let ancestors_a: HashSet<Key> = find_ancestors(a, depth).into_iter()
                                                            .collect();

    let ancestors_b: HashSet<Key> = find_ancestors(b, depth).into_iter()
                                                            .collect();
    let na = find_finest_common_ancestor(a, b, depth);


    let mut working_list: HashSet<Key> = find_children(&na, depth).into_iter()
                                                                  .collect();
    let mut minimal_tree: Keys = Vec::new();

    loop {

        let mut aux_list: HashSet<Key> = HashSet::new();
        let mut len = 0;

        for w in &working_list {

            if ((a < w) & (w < b)) & !ancestors_b.contains(w) {
                aux_list.insert(w.clone());
                len += 1;
            } else if ancestors_a.contains(w) | ancestors_b.contains(w) {
                for child in find_children(w, depth) {
                    aux_list.insert(child);
                }
            }
        }

        if len == working_list.len() {
            minimal_tree = aux_list.into_iter()
                                   .collect();
            break
        } else {
            working_list = aux_list;
        }
    }

    minimal_tree.sort();
    minimal_tree
}


/// Find coarsest 'seeds' at each processor. These are used to seed
/// The construction of a minimal block octree in Algorithm 4 of Sundar et. al.
pub fn find_seeds(
    local_leaves: &Keys,
    depth: &u64
) -> Keys {

    // Find least and greatest leaves on processor
    let min: Key = local_leaves.iter().min().unwrap().clone();
    let max: Key = local_leaves.iter().max().unwrap().clone();

    // Complete region between least and greatest leaves
    let complete = complete_region(&min, &max, depth);

    // Find blocks
    let levels: Vec<u64> = complete.iter().map(|k| {k.3}).collect();

    let mut coarsest_level = depth.clone();

    for l in levels {
        if l < coarsest_level {
            coarsest_level = l;
        }
    }

    let seed_idxs: Vec<usize> = complete.iter()
                                         .enumerate()
                                         .filter(|&(_, &value)| value.3 == coarsest_level)
                                         .map(|(index, _)| index)
                                         .collect();

    let seeds: Keys = seed_idxs.iter()
                                 .map(|&i| complete[i])
                                 .collect();

    seeds
}


/// Remove overlaps from a sorted list of octants, algorithm 7 in Sundar et. al.
pub fn linearise(keys: &Keys, depth: &u64) -> Keys {

    let mut linearised: Keys = Vec::new();
    for i in 0..(keys.len()-1) {
        let curr = keys[i];
        let next = keys[i+1];
        let ancestors_next: HashSet<Key> = find_ancestors(&next, depth).into_iter()
                                                                       .collect();
        if !ancestors_next.contains(&curr) {
            linearised.push(curr)
        }
    }
    linearised
}


/// Complete a distributed blocktree from the seed octants, following Algorithm 4
/// in Sundar. et. al.
pub fn complete_blocktree(
    seeds: &mut Keys,
    depth: &u64,
    rank: i32,
    nprocs: u16,
    world: SystemCommunicator
) -> Keys {
    if rank == 0 {
        let root = Key(0, 0, 0, 0);
        let dfd_root = find_deepest_first_descendent(&root, &depth);
        let min = seeds.iter().min().unwrap();
        let na = find_finest_common_ancestor(&dfd_root, min, &depth);
        let mut first_child = na.clone();
        first_child.3 += 1;
        seeds.push(first_child);
        seeds.sort();
    }

    if rank == (nprocs-1).into() {
        let root = Key(0, 0, 0, 0);
        let dld_root = find_deepest_last_descendent(&root, &depth);
        let max = seeds.iter().max().unwrap();
        let na = find_finest_common_ancestor(&dld_root, max, &depth);
        let children = find_children(&na, &depth);
        let last_child = children.iter().max().unwrap().clone();
        seeds.push(last_child);
    }

    // Send required data to partner process.
    if rank > 0 {
        let min = seeds.iter().min().unwrap().clone();
        world.process_at_rank(rank-1).send(&min);
        // println!("sending {:?} at rank {:?}", min, rank);
    }

    if rank < (nprocs-1).into() {
        let rec = world.any_process().receive::<Key>();
        seeds.push(rec.0);
        // println!("Receieved {:?} at rank {} ", rec, rank);
    }

    // Complete region between seeds at each process
    let mut local_blocktree: Keys = Vec::new();

    for i in 0..(seeds.len()-1) {
        let a = seeds[i];
        let b = seeds[i+1];

        let mut tmp = complete_region(&a, &b, &depth);
        local_blocktree.push(a);
        local_blocktree.append(&mut tmp);
    }

    if rank == (nprocs-1).into() {
        local_blocktree.push(seeds.last().unwrap().clone());
    }

    local_blocktree.sort();
    local_blocktree
}


pub fn find_block_weights(
    local_leaves: &Keys,
    local_blocktree: &Keys,
    depth: &u64,
) -> Weights {

    let local_leaves_set: HashSet<Key> = local_leaves.into_iter()
                                                     .cloned()
                                                     .collect();

    let mut weights: Weights = vec![Weight(0); local_blocktree.len()];

    for (i, block) in local_blocktree.iter().enumerate() {
        let level = block.3;
        let descendents = find_descendents(&block, &level, &depth);

        let mut w = 0;

        for d in descendents {
            if local_leaves_set.contains(&d) {
                w += 1;
            }
        }
        weights[i] = Weight(w)
    }
    weights
}


/// Re-partition the blocks so that amount of computation on
/// each node is balanced
pub fn block_partition(
    weights: Weights,
    local_blocktree: &mut Keys,
    nprocs: u16,
    rank: i32,
    size: i32,
    world: SystemCommunicator,
) {

    let mut local_weight = weights.iter().fold(0, |acc, x| acc + x.0);
    let mut local_nblocks = local_blocktree.len();
    let mut cumulative_weight = 0;
    let mut cumulative_nblocks = 0;
    let mut total_weight = 0;
    let mut total_nblocks = 0;
    world.scan_into(&local_weight, &mut cumulative_weight, &SystemOperation::sum());
    world.scan_into(&local_nblocks, &mut cumulative_nblocks, &SystemOperation::sum());

    // Broadcast total weight from last process
    let last_rank: Rank = (nprocs-1) as Rank;
    let last_process = world.process_at_rank(last_rank);

    if rank == last_rank {
        total_weight = cumulative_weight.clone();
        total_nblocks = cumulative_nblocks.clone();
    } else {
        total_weight = 0;
        total_nblocks = 0;
    }

    last_process.broadcast_into(&mut total_weight);
    last_process.broadcast_into(&mut total_nblocks);

    // Maximum weight per process
    let w: u64 = (total_weight as f64 / nprocs as f64).ceil() as u64;
    let k: u64 = total_weight % (nprocs as u64);

    let mut local_cumulative_weights = weights.clone();
    let mut sum = 0;
    for (i, w) in weights.iter().enumerate() {
        sum += w.0;
        local_cumulative_weights[i] = Weight(sum+cumulative_weight-local_weight as u64)
    }

        let p: u64= (rank+1) as u64;
    let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
    let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };

    let mut q: Keys = Vec::new();

    if p <= k{
        let cond1: u64 = (p-1)*((w+1));
        let cond2: u64 = p*((w+1));

        for (i, &block) in local_blocktree.iter().enumerate() {
            if  (cond1 <= local_cumulative_weights[i].0)
                & (local_cumulative_weights[i].0 < cond2)
            {
                q.push(block);
            }
        }
    } else {
        let cond1: u64 = (p-1)*w + k;
        let cond2: u64 = p*w + k;

        for (i, &block) in local_blocktree.iter().enumerate() {
            if (cond1 <= local_cumulative_weights[i].0)
                & (local_cumulative_weights[i].0 < cond2)
            {
                q.push(block);
            }
        }
    }

    // Send receive qs with partner process
    let next_process = world.process_at_rank(next_rank);
    let previous_process = world.process_at_rank(previous_rank);

    let mut received_blocks: Keys = vec![Key(0, 0, 0, SENTINEL); total_nblocks];

    p2p::send_receive_into(&q[..], &previous_process, &mut received_blocks[..], &next_process);

    received_blocks = received_blocks.iter()
                                     .filter(|b| b.3 != SENTINEL)
                                     .cloned()
                                     .collect();


    for sent in q {
        local_blocktree.iter()
                       .position(|&n| n == sent)
                       .map(|e| local_blocktree.remove(e));
    }

    local_blocktree.extend(&received_blocks);

}