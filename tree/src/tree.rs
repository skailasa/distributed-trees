
use std::collections::HashSet;

use memoffset::offset_of;
use mpi::{
    collective::SystemOperation,
    datatype::{Equivalence, UncommittedUserDatatype, UserDatatype},
    point_to_point as p2p,
    topology::{Rank, SystemCommunicator},
    traits::*,
    Address,
};
use rand::{thread_rng, Rng};

use crate::morton::{
    MAX_POINTS, Key, Keys, Leaf, Leaves, find_ancestors, find_children,
    find_deepest_first_descendent, find_deepest_last_descendent, find_finest_common_ancestor,
};

/// Sample density for oversampled parallel Sample Sort implementation.
const K: usize = 10;

/// Null process marker for MPI functions.
pub const MPI_PROC_NULL: i32 = -1;

#[derive(Debug, Copy, Clone)]
/// **Weight** of a given **Block**. Defined by number of original **Leaf** nodes it contains.
pub struct Weight(pub u64);
/// Vector of **Weights**.
pub type Weights = Vec<Weight>;

unsafe impl Equivalence for Weight {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1],
            &[offset_of!(Weight, 0) as Address],
            &[UncommittedUserDatatype::contiguous(1, &u64::equivalent_datatype()).as_ref()],
        )
    }
}

/// Adapted from algorithm 3 in [1]. Construct a minimal octree between two octants.
pub fn complete_region(a: &Key, b: &Key, depth: &u64) -> Keys {
    let ancestors_a: HashSet<Key> = find_ancestors(a, depth).into_iter().collect();

    let ancestors_b: HashSet<Key> = find_ancestors(b, depth).into_iter().collect();
    let na = find_finest_common_ancestor(a, b, depth);

    let mut working_list: HashSet<Key> = find_children(&na, depth).into_iter().collect();
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
            minimal_tree = aux_list.into_iter().collect();
            break;
        } else {
            working_list = aux_list;
        }
    }

    minimal_tree.sort();
    minimal_tree
}

/// Find unique **Leaves**, and merge together their point sets.
pub fn unique_leaves(mut leaves: Leaves, sorted: bool) -> Leaves {
    // Container for result
    let mut unique: Leaves = Vec::new();

    // Sort leaves
    if !sorted {
        leaves.sort();
    }

    let mut leaf_indices: Vec<usize> = Vec::new();
    let mut nunique = 0;
    let mut curr = Leaf::default();

    for (i, leaf) in leaves.iter().enumerate() {
        if curr != *leaf {
            curr = leaf.clone();
            leaf_indices.push(i);
            nunique += 1;
        }
    }

    leaf_indices.push(leaves.len());

    for i in 0..nunique {
        let lidx = leaf_indices[i as usize];
        let ridx = leaf_indices[(i + 1) as usize];

        // Pick out leaves in this range, and combine their points
        let mut acc = leaves[lidx].clone();

        let mut points_idx = acc.npoints() as usize;

        for j in (lidx + 1)..ridx {
            let npoints = leaves[j].npoints() as usize;

            for k in 0..npoints {
                if (points_idx + k) >= MAX_POINTS {
                    panic!(
                        "You are packing too many points into leaf,
                        you need to increase the depth of your tree!"
                    )
                }

                acc.points.0[points_idx + k] = leaves[j].points.0[k];
            }

            points_idx += npoints;
        }
        unique.push(acc);
    }
    unique
}

/// Find coarsest **Seeds** at each processor. These are used to seed the construction of a minimal
/// block octree in Algorithm 4 of [1].
pub fn find_seeds(local_leaves: &Leaves, depth: &u64) -> Keys {
    // Find least and greatest leaves on processor
    let min: Key = local_leaves.iter().min().unwrap().key.clone();
    let max: Key = local_leaves.iter().max().unwrap().key.clone();

    // Complete region between least and greatest leaves
    let complete = complete_region(&min, &max, depth);

    // Find blocks
    let levels: Vec<u64> = complete.iter().map(|k| k.3).collect();

    let mut coarsest_level = depth.clone();

    for l in levels {
        if l < coarsest_level {
            coarsest_level = l;
        }
    }

    let seed_idxs: Vec<usize> = complete
        .iter()
        .enumerate()
        .filter(|&(_, &value)| value.3 == coarsest_level)
        .map(|(index, _)| index)
        .collect();

    let seeds: Keys = seed_idxs.iter().map(|&i| complete[i]).collect();

    seeds
}

/// Transfer leaves based on **Seeds**. After distributed coarse block octree is found, leaves
/// smaller than the minimum **Seed** on  a given processor must be handed to its partner from
/// algorithm 4 of [1].
pub fn transfer_leaves_to_coarse_blocktree(
    local_leaves: &Leaves,
    seeds: &Keys,
    rank: Rank,
    world: SystemCommunicator,
    nprocs: i32,
) -> Leaves {
    let mut min_seed = Key::default();
    if rank == 0 {
        min_seed = local_leaves.iter().min().unwrap().key.clone();
    } else {
        min_seed = seeds.iter().min().unwrap().clone();
    }

    let mut received: Leaves = Vec::new();

    let prev_rank = rank - 1;

    if rank > 0 {
        let msg: Leaves = local_leaves
            .iter()
            .filter(|&l| &l.key < &min_seed)
            .cloned()
            .collect();

        world.process_at_rank(prev_rank).send(&msg[..]);
    }

    if rank < (nprocs as i32 - 1) {
        let (mut rec, _) = world.any_process().receive_vec::<Leaf>();
        received.append(&mut rec);
    }

    let mut local_leaves: Leaves = local_leaves
        .iter()
        .filter(|&l| &l.key >= &min_seed)
        .cloned()
        .collect();

    local_leaves.append(&mut received);
    local_leaves.sort();
    local_leaves
}

/// Remove overlaps from a list of octants, algorithm 7 in [1].
pub fn linearise(keys: &mut Keys, depth: &u64, sorted: bool) -> Keys {

    if !sorted {
        keys.sort();
    }

    let mut linearised: Keys = Vec::new();
    for i in 0..(keys.len() - 1) {
        let curr = keys[i];
        let next = keys[i + 1];
        let ancestors_next: HashSet<Key> = find_ancestors(&next, depth).into_iter().collect();
        if !ancestors_next.contains(&curr) {
            linearised.push(curr)
        }
    }
    linearised
}

/// Complete a distributed blocktree from the seed octants, algorithm 4 in [1].
pub fn complete_blocktree(
    seeds: &mut Keys,
    depth: &u64,
    rank: i32,
    nprocs: u16,
    world: SystemCommunicator,
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

    if rank == (nprocs - 1).into() {
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
        world.process_at_rank(rank - 1).send(&min);
    }

    if rank < (nprocs - 1).into() {
        let rec = world.any_process().receive::<Key>();
        seeds.push(rec.0);
    }

    // Complete region between seeds at each process
    let mut local_blocktree: Keys = Vec::new();

    for i in 0..(seeds.len() - 1) {
        let a = seeds[i];
        let b = seeds[i + 1];

        let mut tmp = complete_region(&a, &b, &depth);
        local_blocktree.push(a);
        local_blocktree.append(&mut tmp);
    }

    if rank == (nprocs - 1).into() {
        local_blocktree.push(seeds.last().unwrap().clone());
    }

    local_blocktree.sort();
    local_blocktree
}

/// Associate a given set of **Blocks** with a given set of **Leaves**.
pub fn assign_blocks_to_leaves(local_leaves: &mut Leaves, local_blocktree: &Keys, depth: &u64) {
    let local_blocktree_set: HashSet<Key> = local_blocktree.iter().cloned().collect();

    for leaf in local_leaves.iter_mut() {
        let ancestors = find_ancestors(&leaf.key, depth);
        for ancestor in ancestors {
            if local_blocktree_set.contains(&ancestor) {
                leaf.block = ancestor.clone();
                break;
            }
        }
    }
}

/// Find the **Weights** of a given set of **Blocks**.
pub fn find_block_weights(leaves: &Leaves, blocktree: &Keys) -> Weights {
    let mut weights: Weights = Vec::new();

    for &block in blocktree.iter() {
        let counts: u64 = leaves.iter().filter(|&l| l.block == block).count() as u64;
        weights.push(Weight(counts));
    }
    weights
}

/// Transfer **Leaves** to correspond to the final load balanced blocktree.
pub fn transfer_leaves_to_final_blocktree(
    sent_blocks: &Keys,
    mut local_leaves: Leaves,
    nprocs: u16,
    rank: Rank,
    world: SystemCommunicator,
) -> Leaves {
    let mut received: Leaves = Vec::new();
    let mut msg: Leaves = Vec::new();

    for &block in sent_blocks.iter() {
        let mut to_send: Leaves = local_leaves
            .iter()
            .filter(|&l| l.block == block)
            .cloned()
            .collect();
        msg.append(&mut to_send);
    }

    // Remove these leaves from the local leaves
    for &block in sent_blocks.iter() {
        local_leaves = local_leaves
            .iter()
            .filter(|&l| l.block != block)
            .cloned()
            .collect();
    }

    for i in 0..((nprocs) as i32) {
        if i + 1 < ((nprocs) as i32) {
            if rank == (i + 1) {
                world.process_at_rank(i).send_with_tag(&msg[..], rank);
            }
            if rank == i {
                let (mut rec, _) = world.any_process().receive_vec::<Leaf>();
                received.append(&mut rec);
            }
        } else {
            if rank == 0 {
                world.process_at_rank(i).send_with_tag(&msg[..], rank);
            }
            if rank == i {
                let (mut rec, _) = world.any_process().receive_vec::<Leaf>();
                received.append(&mut rec);
            }
        }
    }

    // Append received leaves
    local_leaves.append(&mut received);
    local_leaves
}

/// Re-partition the blocks so that amount of computation on each node is balanced. Return mapping
/// between block and rank to which it was sent.
pub fn block_partition(
    weights: Weights,
    local_blocktree: &mut Keys,
    nprocs: u16,
    rank: i32,
    size: i32,
    world: SystemCommunicator,
) -> Keys {
    let local_weight = weights.iter().fold(0, |acc, x| acc + x.0);
    let local_nblocks = local_blocktree.len();
    let mut cumulative_weight = 0;
    let mut cumulative_nblocks = 0;
    let mut total_weight = 0;
    let mut total_nblocks = 0;
    world.scan_into(
        &local_weight,
        &mut cumulative_weight,
        &SystemOperation::sum(),
    );
    world.scan_into(
        &local_nblocks,
        &mut cumulative_nblocks,
        &SystemOperation::sum(),
    );

    // Broadcast total weight from last process
    let last_rank: Rank = (nprocs - 1) as Rank;
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
        local_cumulative_weights[i] = Weight(sum + cumulative_weight - local_weight as u64)
    }

    let p: u64 = (rank + 1) as u64;
    let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
    let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };

    let mut q: Keys = Vec::new();

    if p <= k {
        let cond1: u64 = (p - 1) * (w + 1);
        let cond2: u64 = p * (w + 1);

        for (i, &block) in local_blocktree.iter().enumerate() {
            if (cond1 <= local_cumulative_weights[i].0) & (local_cumulative_weights[i].0 < cond2) {
                q.push(block);
            }
        }
    } else {
        let cond1: u64 = (p - 1) * w + k;
        let cond2: u64 = p * w + k;

        for (i, &block) in local_blocktree.iter().enumerate() {
            if (cond1 <= local_cumulative_weights[i].0) & (local_cumulative_weights[i].0 < cond2) {
                q.push(block);
            }
        }
    }

    // Send receive qs with partner process
    let next_process = world.process_at_rank(next_rank);
    let previous_process = world.process_at_rank(previous_rank);

    let mut received_blocks: Keys = vec![Key::default(); total_nblocks];

    p2p::send_receive_into(
        &q[..],
        &previous_process,
        &mut received_blocks[..],
        &next_process,
    );

    received_blocks = received_blocks
        .iter()
        .filter(|&&b| b != Key::default())
        .cloned()
        .collect();

    for sent in &q {
        local_blocktree
            .iter()
            .position(|&n| n == *sent)
            .map(|e| local_blocktree.remove(e));
    }

    local_blocktree.extend(&received_blocks);

    q
}

/// Perform parallelised sample sort on a distributed set of **Leaves**.
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


mod tests {
    use super::*;

    use crate::morton::{Point, PointsArray, MAX_POINTS};

    #[test]
    #[should_panic]
    fn test_unique_panic() {
        // Test that you cannot overpack a unique leaf when merging leaves.
        let mut leaves: Leaves = vec![
            Leaf {
                key: Key(0, 0, 0, 1),
                block: Key::default(),
                points: PointsArray([Point::default(); MAX_POINTS]),
            },
            Leaf {
                key: Key(0, 0, 0, 1),
                block: Key::default(),
                points: PointsArray([Point::default(); MAX_POINTS]),
            },
            Leaf {
                key: Key(0, 0, 0, 1),
                block: Key::default(),
                points: PointsArray([Point::default(); MAX_POINTS]),
            },
            Leaf {
                key: Key(0, 0, 0, 1),
                block: Key::default(),
                points: PointsArray([Point::default(); MAX_POINTS]),
            },
        ];

        for mut leaf in &mut leaves {
            for i in 0..150 {
                leaf.points.0[i] = Point(0., 0., 0.);
            }
        }

        let unique = unique_leaves(leaves, true);
    }

    #[test]
    fn test_unique() {
        let mut leaves: Leaves = vec![
            Leaf {
                key: Key(0, 0, 0, 1),
                block: Key::default(),
                points: PointsArray([Point::default(); MAX_POINTS]),
            },
            Leaf {
                key: Key(0, 0, 0, 1),
                block: Key::default(),
                points: PointsArray([Point::default(); MAX_POINTS]),
            },
            Leaf {
                key: Key(0, 0, 0, 1),
                block: Key::default(),
                points: PointsArray([Point::default(); MAX_POINTS]),
            },
            Leaf {
                key: Key(0, 0, 0, 1),
                block: Key::default(),
                points: PointsArray([Point::default(); MAX_POINTS]),
            },
        ];

        for mut leaf in &mut leaves {
            for i in 0..4 {
                leaf.points.0[i] = Point(0., 0., 0.);
            }
        }

        let unique = unique_leaves(leaves, true);

        assert_eq!(unique[0].npoints(), 16)
    }
}
