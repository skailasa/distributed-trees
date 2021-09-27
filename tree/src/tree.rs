use std::collections::HashSet;

use memoffset::offset_of;
use mpi::{
    Address,
    datatype::{Equivalence, UserDatatype, UncommittedUserDatatype, Partition},
    Count,
    point_to_point as p2p,
    traits::*,
    topology::{Process, SystemCommunicator},
};

use crate::morton::{
    Key,
    Keys,
    find_ancestors,
    find_children,
    find_finest_common_ancestor,
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

/// Merge two sorted vectors of Morton keys
///
/// # Arguments
/// `a` - Sorted vector of Morton keys
/// `b` - Sorted vector of Morton keys
pub fn merge(a: &Keys, b: &Keys) -> Vec<Key> {

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


/// Find coarsest 'blocks' at each processor. These are used to seed
/// The construction of a minimal octree in Algorithm 4 of Sundar et. al.
pub fn find_blocks(
    rank: i32,
    mut local_leaves: Keys,
    depth: &u64
) -> Keys {

    // Find least and greatest leaves on processor
    let min: Key = local_leaves.iter().min().unwrap().clone();
    let max: Key = local_leaves.iter().max().unwrap().clone();

    // let a =  min;
    // let b = max;
    // println!("ab{} = [", rank);
    // println!("np.array([{}, {}, {}, {}]),", a.0, a.1, a.2, a.3);
    // println!("np.array([{}, {}, {}, {}])", b.0, b.1, b.2, b.3);
    // println!("]");

    // Complete region between least and greatest leaves
    let complete = complete_region(&min, &max, depth);

    // println!("complete{} = [", rank);
    // for node in &complete {
    //     println!("np.array([{}, {}, {}, {}], dtype=np.int64),", node.0, node.1, node.2, node.3);
    // };
    // println!("]");
    // println!(" ");

    // Find blocks
    let levels: Vec<u64> = complete.iter()
                        .map(|k| {k.3})
                        .collect();

    let mut coarsest_level = depth.clone();

    for l in levels {
        if l < coarsest_level {
            coarsest_level = l;
        }
    }

    let block_idxs: Vec<usize> = complete.iter()
                                    .enumerate()
                                    .filter(|&(_, &value)| value.3 == coarsest_level)
                                    .map(|(index, _)| index)
                                    .collect();

    let blocks: Keys = block_idxs.iter()
                               .map(|&i| complete[i])
                               .collect();

    blocks
}
