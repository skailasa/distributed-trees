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
    nearest_common_ancestor,
};

pub const MPI_PROC_NULL: i32 = -1;

/// Implement MPI equivalent datatype for Morton keys
unsafe impl Equivalence for Key {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1],
            &[
                offset_of!(Key, 0) as Address,
            ],
            &[
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

pub struct Leaves {
    pub local: Keys,
    pub received: Keys,
}


pub fn nleaves(depth: u16) -> u16 {
    let base: u16 = 2;
    let dim: u16 = 3;
    u16::pow(base, (dim*depth).into())
}

/// Distribute unsorted leaves across processes
pub fn distribute_leaves(
    tree: Keys,
    world: SystemCommunicator,
    rank: i32,
    root_rank: i32,
    root_process: Process<SystemCommunicator>,
    nprocs: u16,
) -> Leaves {
    let nleaves = tree.len() as u16;

    let load = balance_load(nleaves, nprocs);
    let bufsize = load.counts[rank as usize];
    let max_bufsize = load.counts[0];

    // Buffer for keys local to this process
    let mut local_leaves = vec![Key(0); bufsize as usize];

    // Buffer for receiving partner keys
    let received_leaves = vec![Key(0); max_bufsize as usize];

    // Distribute the leaves to all participating processes
    if rank == root_rank {

        // Calculate load
        let partition = Partition::new(&tree[..], load.counts, load.displs);

        // ScatterV
        root_process.scatter_varcount_into_root(&partition, &mut local_leaves[..])

    } else {
        root_process.scatter_varcount_into(&mut local_leaves[..])
    }

    // Ensure that scatter has happened
    world.barrier();

    // Sort local leaves using quicksort
    local_leaves.sort();

    Leaves {
        local: local_leaves,
        received: received_leaves,
    }
}

/// Sort distributed leaves in Morton order
pub fn parallel_morton_sort(
    mut leaves: Leaves,
    world: SystemCommunicator,
    rank: i32,
    nprocs: u16,
) {

    // Guaranteed to converge in nprocs phases
    for phase in 0..nprocs {
        // println!("phase {}", phase);
        let partner = compute_partner(rank, phase as i32, nprocs as i32);

        // Send local leaves to partner, and recieve their leaves
        if partner != MPI_PROC_NULL {

            let partner_process = world.process_at_rank(partner);

            p2p::send_receive_into(&leaves.local[..], &partner_process, &mut leaves.received[..], &partner_process);
            // println!("Rank {}, received: {:?}  sent {:?}", rank, received_leaves, local_leaves);

            // Perform merge
            let merged = merge(&leaves.local, &leaves.received);

            let mid = middle(merged.len());

            if rank < partner {
                // Keep smaller keys
                leaves.local = merged[..mid].to_vec();

            } else {
                // Keep larger keys
                leaves.local = merged[mid..].to_vec();
            }
        }
    }

    println!("rank {}", rank);
    println!("[");
    for leaf in leaves.local {
        println!("{}", leaf);
    }
    println!("]\n");
}


/// Construct a minimal linear octree between two octants
/// Algorithm 3 in Sundar et. al.
pub fn complete_region(a: &Key, b: &Key) -> Keys {

    let ancestors_a: HashSet<Key> = find_ancestors(a).into_iter()
                                                     .collect();
    let ancestors_b: HashSet<Key> = find_ancestors(b).into_iter()
                                                     .collect();
    let na = nearest_common_ancestor(a, b);


    let mut working_list: HashSet<Key> = find_children(&na).into_iter()
                                                           .collect();
    let mut minimal_tree: Keys = Vec::new();

    let mut i = 0;

    loop {

        let mut aux_list: HashSet<Key> = HashSet::new();
        let mut len = 0;

        // println!("len working list {}", working_list.len());
        // println!("before [");
        // for node in &working_list {
        //     println!("{}", node);
        // }
        // println!("]\n");
        for w in &working_list {

            if ((a < w) & (w < b)) & (ancestors_b.contains(w) == false) {
                aux_list.insert(w.clone());
                len += 1;
            } else if ancestors_a.contains(w) | ancestors_b.contains(w) {
                for child in find_children(w) {
                    aux_list.insert(child);
                }
            }
        }
        // println!("after [");
        // for node in &aux_list {
        //     println!("{}", node);
        // }
        // println!("]\n");

        if len == working_list.len() {
            minimal_tree = aux_list.into_iter()
                                   .collect();
            break
        } else {
            working_list = aux_list;
        }

        i += 1;
    }

    minimal_tree.sort();
    minimal_tree
}
