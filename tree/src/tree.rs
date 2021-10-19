use std::collections::{HashMap, HashSet};

use memoffset::offset_of;
use mpi::{
    collective::SystemOperation,
    datatype::{Equivalence, UncommittedUserDatatype, UserDatatype},
    environment::Universe,
    topology::{Rank, SystemCommunicator},
    traits::*,
    Address,
};
use rand::{thread_rng, Rng};

use crate::morton::{
    encode_points, find_ancestors, find_children, find_deepest_first_descendent,
    find_deepest_last_descendent, find_finest_common_ancestor, keys_to_leaves, Key, Keys, Leaf,
    Leaves, Point, Points,
};

/// Sample density for over sampled parallel Sample Sort implementation.
const K: usize = 10;

/// Null process marker for MPI functions.
pub const MPI_PROC_NULL: i32 = -1;

/// Type alias for a tree data structure.
pub type Tree = HashMap<Key, Leaves>;

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

/// Adapted from algorithm 3 in [1]. Construct a minimal octree between two octants, excluding the
/// two octants (sequential).
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
                aux_list.insert(*w);
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

/// Make **Leaves** unique, check that they don't exceed 'ncrit' points per leaf (sequential).
pub fn unique_leaves(mut leaves: Leaves, ncrit: &usize, sorted: bool) -> Leaves {
    // Container for result
    let mut unique: Leaves = Vec::new();

    // Sort leaves
    if !sorted {
        leaves.sort();
    }

    // Find number of unique leaves, and their indices
    let mut curr_idx: usize = 0;
    let mut curr = leaves[curr_idx].clone();
    unique.push(curr);

    for &leaf in leaves.iter().skip(1) {
        if curr != leaf {
            unique.push(curr);
            curr_idx += 1;
            curr = leaf;
        } else {
            let n = unique[curr_idx].npoints + leaf.npoints;
            if n <= *ncrit {
                unique[curr_idx].npoints += leaf.npoints;
            } else {
                panic!("You are packing too many points into a leaf, increase tree depth!")
            }
        }
    }
    unique
}

/// Find coarsest **Seeds** at each processor. These are used to seed the construction of a minimal
/// block octree in Algorithm 4 of [1] (sequential).
pub fn find_seeds(local_leaves: &[Leaf], depth: &u64) -> Keys {
    // Find least and greatest leaves on processor
    let min: Key = local_leaves.iter().min().unwrap().key;
    let max: Key = local_leaves.iter().max().unwrap().key;

    // Complete region between least and greatest leaves
    let mut complete = complete_region(&min, &max, depth);
    complete.push(min);
    complete.push(max);

    // Find blocks
    let levels: Vec<u64> = complete.iter().map(|k| k.3).collect();

    let mut coarsest_level = *depth;

    for l in levels {
        if l < coarsest_level {
            coarsest_level = l;
        }
    }

    let mut seed_idxs: Vec<usize> = Vec::new();

    for (i, node) in complete.iter().enumerate() {
        if node.3 == coarsest_level {
            seed_idxs.push(i);
        }
    }

    let seeds: Keys = seed_idxs.iter().map(|&i| complete[i as usize]).collect();

    seeds
}

/// Transfer leaves based on **Seeds**. After distributed coarse block octree is found, leaves
/// smaller than the minimum **Seed** on  a given processor must be handed to its partner from
/// algorithm 4 of [1] (parallel).
pub fn transfer_leaves_to_coarse_blocktree(
    points: &[Point],
    local_leaves: &[Leaf],
    received_points: &mut Points,
    received_leaves: &mut Leaves,
    seeds: &[Key],
    rank: Rank,
    world: SystemCommunicator,
    size: Rank,
) {
    let mut min_seed = Key::default();
    if rank == 0 {
        min_seed = local_leaves.iter().min().unwrap().key;
    } else {
        min_seed = *seeds.iter().min().unwrap();
    }

    let prev_rank = rank - 1;

    if rank > 0 {
        let msg: Leaves = local_leaves
            .iter()
            .filter(|&l| l.key < min_seed)
            .cloned()
            .collect();

        world.process_at_rank(prev_rank).send(&msg[..]);
    }

    if rank < (size - 1) {
        let (mut rec, _) = world.any_process().receive_vec::<Leaf>();
        received_leaves.append(&mut rec);
    }

    if rank > 0 {
        let msg: Points = points
            .iter()
            .filter(|&l| l.key < min_seed)
            .cloned()
            .collect();

        world.process_at_rank(prev_rank).send(&msg[..]);
    }

    if rank < (size - 1) {
        let (mut rec, _) = world.any_process().receive_vec::<Point>();
        received_points.append(&mut rec);
    }

    let mut local_leaves: Leaves = local_leaves
        .iter()
        .filter(|&l| l.key >= min_seed)
        .cloned()
        .collect();

    let mut points: Points = points
        .iter()
        .filter(|&l| l.key >= min_seed)
        .cloned()
        .collect();

    received_leaves.append(&mut local_leaves);
    received_points.append(&mut points);

    // Re Sort received leaves
    received_leaves.sort();
}

/// Remove overlaps from a list of octants, algorithm 7 in [1], expects input keys to be sorted
/// (sequential).
pub fn linearise(keys: &mut Keys, depth: &u64) -> Keys {
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

/// Complete a distributed blocktree from the seed octants, algorithm 4 in [1] (parallel).
pub fn complete_blocktree(
    seeds: &mut Keys,
    depth: &u64,
    rank: Rank,
    size: Rank,
    world: SystemCommunicator,
) -> Keys {
    if rank == 0 {
        let root = Key(0, 0, 0, 0);
        let dfd_root = find_deepest_first_descendent(&root, depth);
        let min = seeds.iter().min().unwrap();
        let na = find_finest_common_ancestor(&dfd_root, min, depth);
        let mut first_child = na;
        first_child.3 += 1;
        seeds.push(first_child);
        seeds.sort();
    }

    if rank == (size - 1) {
        let root = Key(0, 0, 0, 0);
        let dld_root = find_deepest_last_descendent(&root, depth);
        let max = seeds.iter().max().unwrap();
        let na = find_finest_common_ancestor(&dld_root, max, depth);
        let children = find_children(&na, depth);
        let last_child = *children.iter().max().unwrap();
        seeds.push(last_child);
    }

    // Send required data to partner process.
    if rank > 0 {
        let min = *seeds.iter().min().unwrap();
        world.process_at_rank(rank - 1).send(&min);
    }

    if rank < (size - 1) {
        let rec = world.any_process().receive::<Key>();
        seeds.push(rec.0);
    }

    // Complete region between seeds at each process
    let mut local_blocktree: Keys = Vec::new();

    for i in 0..(seeds.len() - 1) {
        let a = seeds[i];
        let b = seeds[i + 1];

        let mut tmp = complete_region(&a, &b, depth);
        local_blocktree.push(a);
        local_blocktree.append(&mut tmp);
    }

    if rank == (size - 1) {
        local_blocktree.push(*seeds.last().unwrap());
    }

    local_blocktree.sort();
    local_blocktree
}

/// Associate a given set of **Blocks** with a given set of **Leaves** (sequential).
pub fn assign_blocks_to_leaves(local_leaves: &mut Leaves, local_blocktree: &[Key], depth: &u64) {
    let local_blocktree_set: HashSet<Key> = local_blocktree.iter().cloned().collect();

    for leaf in local_leaves.iter_mut() {
        let ancestors = find_ancestors(&leaf.key, depth);
        for ancestor in ancestors {
            if local_blocktree_set.contains(&ancestor) {
                leaf.block = ancestor;
                break;
            } else if local_blocktree.contains(&leaf.key) {
                leaf.block = leaf.key;
                break;
            }
        }
    }
}

/// Find the **Weights** of a given set of **Blocks** (sequential).
pub fn find_block_weights(leaves: &[Leaf], blocktree: &[Key]) -> Weights {
    let mut weights: Weights = Vec::new();

    for &block in blocktree.iter() {
        let counts: u64 = leaves.iter().filter(|&l| l.block == block).count() as u64;
        weights.push(Weight(counts));
    }
    weights
}

/// Transfer **Leaves** to correspond to the final load balanced blocktree (parallel).
pub fn transfer_leaves_to_final_blocktree(
    sent_blocks: &[Key],
    mut local_leaves: Leaves,
    size: Rank,
    rank: Rank,
    world: SystemCommunicator,
) -> Leaves {
    let mut received: Leaves = Vec::new();
    let mut msg: Leaves = Vec::new();

    let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
    let prev_rank = if rank > 0 { rank - 1 } else { size - 1 };
    let previous_process = world.process_at_rank(prev_rank);

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
        local_leaves.retain(|l| l.block != block)
    }

    for r in 0..size {
        if r == rank {
            previous_process.send(&msg[..]);
        }
        if r == next_rank {
            let (mut rec, _) = world.any_process().receive_vec::<Leaf>();

            local_leaves.append(&mut rec)
        }
    }

    // Append received leaves
    local_leaves.append(&mut received);
    local_leaves
}

/// Re-partition the blocks so that amount of computation on each node is balanced. Return mapping
/// between block and rank to which it was sent (parallel).
pub fn block_partition(
    weights: Weights,
    local_blocktree: &mut Keys,
    rank: Rank,
    size: Rank,
    world: SystemCommunicator,
) -> Keys {
    let local_weight = weights.iter().fold(0, |acc, x| acc + x.0);
    let local_nblocks = local_blocktree.len();
    let mut cumulative_weight = 0;
    let mut cumulative_nblocks = 0;

    #[allow(unused_variables)]
    let mut total_weight = 0;
    #[allow(unused_variables)]
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
    let last_rank = size - 1;
    let last_process = world.process_at_rank(last_rank);

    if rank == last_rank {
        total_weight = cumulative_weight;
        total_nblocks = cumulative_nblocks;
    } else {
        total_weight = 0;
        total_nblocks = 0;
    }

    last_process.broadcast_into(&mut total_weight);
    last_process.broadcast_into(&mut total_nblocks);

    // Maximum weight per process
    let w: u64 = (total_weight as f64 / size as f64).ceil() as u64;
    let k: u64 = total_weight % (size as u64);

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
    let previous_process = world.process_at_rank(previous_rank);

    let mut received_blocks: Keys = Vec::new();

    for r in 0..size {
        if r == rank {
            previous_process.send(&q[..]);
        }
        if r == next_rank {
            let (mut rec, _) = world.any_process().receive_vec::<Key>();

            received_blocks.append(&mut rec)
        }
    }

    // Remove sent blocks locally, and append received blocks
    for sent in &q {
        local_blocktree
            .iter()
            .position(|&n| n == *sent)
            .map(|e| local_blocktree.remove(e));
    }

    local_blocktree.extend(&received_blocks);

    q
}

/// Split **Blocks** to satisfy a maximum of NCRIT particles per node in the final octree
/// (sequential).
pub fn split_blocks(local_leaves: &mut Leaves, depth: &u64, ncrit: &usize) -> HashMap<Key, Leaves> {
    let mut blocks: HashMap<Key, Leaves> = HashMap::new();

    for &leaf in local_leaves.iter() {
        blocks.entry(leaf.block).or_default().push(leaf);
    }

    loop {
        let mut to_split: Keys = Vec::new();
        for (&key, leaves) in blocks.iter() {
            let mut npoints = 0;
            for leaf in leaves {
                npoints += leaf.npoints;
                if npoints > *ncrit {
                    to_split.push(key);
                    break;
                }
            }
        }

        if to_split.len() > 0 {
            for &key in to_split.iter() {
                let mut leaves = blocks.get(&key).unwrap().clone();
                let children = find_children(&key, depth);
                blocks.remove(&key);
                assign_blocks_to_leaves(&mut leaves, &children, depth);

                for &leaf in leaves.iter() {
                    blocks.entry(leaf.block).or_default().push(leaf);
                }
            }
        } else {
            break;
        }
    }
    blocks
}

/// Perform parallelised sample sort on a distributed set of **Leaves** (parallel).
pub fn sample_sort(
    mut points: &mut Points,
    ncrit: &usize,
    received_leaves: &mut Leaves,
    received_points: &mut Points,
    size: Rank,
    rank: Rank,
    world: SystemCommunicator,
) {
    let local_leaves = keys_to_leaves(&mut points, ncrit);

    let mut received_samples = vec![Leaf::default(); K * (size as usize)];
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
    let mut buckets: Vec<Leaves> = vec![Vec::new(); size as usize];
    // Sort local points into corresponding buckets
    let mut buckets_points: Vec<Points> = vec![Vec::new(); size as usize];

    for leaf in local_leaves.iter() {
        for i in 0..(size as usize) {
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

    for point in points.iter() {
        for i in 0..(size as usize) {
            if i < nsplitters {
                let s = &splitters[i];
                if point.key < s.key {
                    buckets_points[i].push(point.clone());
                    break;
                }
            } else {
                buckets_points[i].push(point.clone())
            }
        }
    }

    // 3. Send all local buckets to their matching processor.
    for r in 0..size {
        if rank != r {
            // let sent_leaves = &buckets[r as usize];
            let sent = &buckets_points[r as usize];
            world.process_at_rank(r).send(&sent[..]);
        } else {
            for _ in 1..world.size() {
                // let (mut rec_leaves, _) = world.any_process().receive_vec::<Leaf>();
                let (mut rec, _) = world.any_process().receive_vec::<Point>();
                received_points.append(&mut rec);
            }
        }
        world.barrier();
    }
    for r in 0..size {
        if rank != r {
            let sent = &buckets[r as usize];
            world.process_at_rank(r).send(&sent[..]);
        } else {
            for _ in 1..world.size() {
                let (mut rec, _) = world.any_process().receive_vec::<Leaf>();
                received_leaves.append(&mut rec);
            }
        }
        world.barrier();
    }
    // 4. Sort leaves on matching processors.
    received_leaves.append(&mut buckets[rank as usize]);
    received_points.append(&mut buckets_points[rank as usize]);
    received_leaves.sort();
}

/// Generate a distributed unbalanced tree from a set of distributed points.
pub fn unbalanced_tree(
    depth: &u64,
    ncrit: &usize,
    universe: Universe,
    mut points: &mut Points,
    x0: Point,
    r0: f64,
) -> Tree {
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    // 1. Encode points to leaf keys inplace.
    encode_points(&mut points, &depth, &depth, &x0, &r0);

    // Temporary buffer for receiving partner keys
    let mut sorted_leaves: Leaves = Vec::new();
    let mut sorted_points: Points = Vec::new();

    // 2. Perform parallel Morton sort over points
    sample_sort(
        &mut points,
        &ncrit,
        &mut sorted_leaves,
        &mut sorted_points,
        size,
        rank,
        world,
    );

    let points = sorted_points;
    let local_leaves = sorted_leaves;

    // 3. Remove duplicates at each processor and remove overlaps if there are any
    let local_leaves = unique_leaves(local_leaves, ncrit, true);

    // 4.i Complete minimal tree on each process, and find seed octants.
    let mut seeds = find_seeds(&local_leaves, depth);

    // 4.ii If leaf is less than the minimum seed in a given process, it needs to be sent to the
    // previous process

    // Temporary buffers to receive points and leaves to coarse block tree.
    let mut received_leaves: Leaves = Vec::new();
    let mut received_points: Points = Vec::new();

    transfer_leaves_to_coarse_blocktree(
        &points,
        &local_leaves,
        &mut received_points,
        &mut received_leaves,
        &seeds,
        rank,
        world,
        size,
    );

    let mut local_leaves = received_leaves;
    let points = received_points;

    // 5. Complete minimal block-tree across processes
    let mut local_blocktree = complete_blocktree(&mut seeds, depth, rank, size, world);

    // Associate leaves with blocks
    assign_blocks_to_leaves(&mut local_leaves, &local_blocktree, depth);

    // 6. Split blocks into adaptive tree, and pass into Octree structure.
    let nodes = split_blocks(&mut local_leaves, depth, ncrit);

    nodes
}

mod tests {
    use super::*;

    use crate::data::random;
    use crate::morton::{find_finest_common_ancestor, find_siblings, Point, MAX_POINTS};

    #[test]
    fn test_complete_region() {
        let a = Key(0, 0, 0, 2);
        let b = Key(3, 3, 3, 2);
        let depth = 2;
        let mut result = complete_region(&a, &b, &depth);
        let fca = find_finest_common_ancestor(&a, &b, &depth);

        let min = result.iter().min().unwrap();
        let max = result.iter().max().unwrap();

        // Test that bounds are satisfied
        assert!(a <= *min);
        assert!(b >= *max);

        // Test that FCA is an ancestor of all nodes in the result
        for node in result.iter() {
            let ancestors = find_ancestors(&node, &depth);
            assert!(ancestors.contains(&fca));
        }
    }

    #[test]
    fn test_unique() {
        let mut leaves: Leaves = vec![
            Leaf {
                key: Key(0, 0, 0, 1),
                block: Key::default(),
                npoints: 11,
            },
            Leaf {
                key: Key(0, 0, 0, 1),
                block: Key::default(),
                npoints: 12,
            },
            Leaf {
                key: Key(0, 0, 0, 1),
                block: Key::default(),
                npoints: 13,
            },
        ];

        let ncrit: usize = 50;

        let unique = unique_leaves(leaves, &ncrit, true);

        assert_eq!(unique[0].npoints, 36)
    }

    fn test_linearise() {
        let key = Key(0, 0, 0, 1);
        let depth = 2;
        let mut children = find_children(&key, &depth);
        children.insert(0, key);

        let linearised = linearise(&mut children, &depth);

        assert!(!linearised.contains(&key));
    }
}
