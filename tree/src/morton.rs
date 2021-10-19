use std::cmp::Ordering;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use memoffset::offset_of;
use mpi::{
    datatype::{Equivalence, UncommittedUserDatatype, UserDatatype},
    Address,
};
use rayon::prelude::*;

/// Maximum points per **Leaf**
pub const MAX_POINTS: usize = 50;

/// Used as an integer sentinel value.
const SENTINEL: KeyType = 999;

type PointType = f64;
#[derive(Clone, Copy, Debug)]
/// **Point**, Cartesian coordinates (x, y, z).
pub struct Point {
    pub x: PointType,
    pub y: PointType,
    pub z: PointType,
    pub key: Key,
    pub global_idx: usize,
}

/// Vector of **Points**.
pub type Points = Vec<Point>;

type KeyType = u64;
#[derive(Clone, Copy, Debug)]
/// **Morton Key**, anchor and level represented as (x, y, z, level).
pub struct Key(pub KeyType, pub KeyType, pub KeyType, pub KeyType);
/// Vector of **Keys**.
pub type Keys = Vec<Key>;

#[derive(Clone, Copy, Debug)]
/// **Leaf Key**, bundles **Morton Key**, associated **Block** and particle **Points** it contains.
pub struct Leaf {
    pub key: Key,
    pub block: Key,
    pub npoints: usize,
}

/// Vector of **Leaves**.
pub type Leaves = Vec<Leaf>;

impl Default for Key {
    fn default() -> Self {
        Key(SENTINEL, SENTINEL, SENTINEL, 0)
    }
}

impl Default for Point {
    fn default() -> Self {
        Point {
            x: PointType::NAN,
            y: PointType::NAN,
            z: PointType::NAN,
            key: Key::default(),
            global_idx: 0,
        }
    }
}

impl Default for Leaf {
    fn default() -> Self {
        Leaf {
            key: Key::default(),
            block: Key::default(),
            npoints: 0,
        }
    }
}

/// Test **Morton Keys** for equality. Keys are considered equal if their anchors and levels match.
fn equal(a: &Key, b: &Key) -> bool {
    (a.0 == b.0) & (a.1 == b.1) & (a.2 == b.2) & (a.3 == b.3)
}

/// Subroutine in less than function, equivalent to comparing floor of log_2(x). Adapted from [3].
fn most_significant_bit(x: u64, y: u64) -> bool {
    (x < y) & (x < (x ^ y))
}

/// Implementation of Algorithm 12 in [1]. to compare the ordering of two **Morton Keys**. If key
/// `a` is less than key `b`, this function evaluates to true.
fn less_than(a: &Key, b: &Key) -> Option<bool> {
    // If anchors match, the one at the coarser level has the lesser Morton id.
    let same_anchor = (a.0 == b.0) & (a.1 == b.1) & (a.2 == b.2);

    match same_anchor {
        true => {
            if a.3 < b.3 {
                Some(true)
            } else {
                Some(false)
            }
        }
        false => {
            let x = vec![a.0 ^ b.0, a.1 ^ b.1, a.2 ^ b.2];

            let mut argmax = 0;

            for dim in 1..3 {
                if most_significant_bit(x[argmax as usize], x[dim as usize]) {
                    argmax = dim
                }
            }

            match argmax {
                0 => {
                    if a.0 < b.0 {
                        Some(true)
                    } else {
                        Some(false)
                    }
                }
                1 => {
                    if a.1 < b.1 {
                        Some(true)
                    } else {
                        Some(false)
                    }
                }
                2 => {
                    if a.2 < b.2 {
                        Some(true)
                    } else {
                        Some(false)
                    }
                }
                _ => None,
            }
        }
    }
}

impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let less = less_than(self, other).unwrap();
        let eq = self.eq(other);

        match eq {
            true => Some(Ordering::Equal),
            false => match less {
                true => Some(Ordering::Less),
                false => Some(Ordering::Greater),
            },
        }
    }
}

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        equal(self, other)
    }
}

impl Eq for Key {}

impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
        self.1.hash(state);
        self.2.hash(state);
        self.3.hash(state);
    }
}

impl Ord for Leaf {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialOrd for Leaf {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let less = less_than(&self.key, &other.key).unwrap();
        let eq = self.eq(other);

        match eq {
            true => Some(Ordering::Equal),
            false => match less {
                true => Some(Ordering::Less),
                false => Some(Ordering::Greater),
            },
        }
    }
}

impl PartialEq for Leaf {
    fn eq(&self, other: &Self) -> bool {
        equal(&self.key, &other.key)
    }
}

impl Eq for Leaf {}

impl Hash for Leaf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.0.hash(state);
        self.key.1.hash(state);
        self.key.2.hash(state);
        self.key.3.hash(state);
    }
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        (self.x == other.x) & (self.y == other.y) & (self.z == other.z)
    }
}

impl Eq for Point {}

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

unsafe impl Equivalence for Leaf {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1, 1, 1],
            &[
                offset_of!(Leaf, key) as Address,
                offset_of!(Leaf, block) as Address,
                offset_of!(Leaf, npoints) as Address,
            ],
            &[
                UncommittedUserDatatype::structured(
                    &[1, 1, 1, 1],
                    &[
                        offset_of!(Key, 0) as Address,
                        offset_of!(Key, 1) as Address,
                        offset_of!(Key, 2) as Address,
                        offset_of!(Key, 3) as Address,
                    ],
                    &[
                        UncommittedUserDatatype::contiguous(1, &u64::equivalent_datatype())
                            .as_ref(),
                        UncommittedUserDatatype::contiguous(1, &u64::equivalent_datatype())
                            .as_ref(),
                        UncommittedUserDatatype::contiguous(1, &u64::equivalent_datatype())
                            .as_ref(),
                        UncommittedUserDatatype::contiguous(1, &u64::equivalent_datatype())
                            .as_ref(),
                    ],
                )
                .as_ref(),
                UncommittedUserDatatype::structured(
                    &[1, 1, 1, 1],
                    &[
                        offset_of!(Key, 0) as Address,
                        offset_of!(Key, 1) as Address,
                        offset_of!(Key, 2) as Address,
                        offset_of!(Key, 3) as Address,
                    ],
                    &[
                        UncommittedUserDatatype::contiguous(1, &u64::equivalent_datatype())
                            .as_ref(),
                        UncommittedUserDatatype::contiguous(1, &u64::equivalent_datatype())
                            .as_ref(),
                        UncommittedUserDatatype::contiguous(1, &u64::equivalent_datatype())
                            .as_ref(),
                        UncommittedUserDatatype::contiguous(1, &u64::equivalent_datatype())
                            .as_ref(),
                    ],
                )
                .as_ref(),
                UncommittedUserDatatype::contiguous(1, &usize::equivalent_datatype()).as_ref(),
            ],
        )
    }
}

unsafe impl Equivalence for Point {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1, 1, 1, 1, 1],
            &[
                offset_of!(Point, x) as Address,
                offset_of!(Point, y) as Address,
                offset_of!(Point, z) as Address,
                offset_of!(Point, key) as Address,
                offset_of!(Point, global_idx) as Address,
            ],
            &[
                UncommittedUserDatatype::contiguous(1, &f64::equivalent_datatype()).as_ref(),
                UncommittedUserDatatype::contiguous(1, &f64::equivalent_datatype()).as_ref(),
                UncommittedUserDatatype::contiguous(1, &f64::equivalent_datatype()).as_ref(),
                UncommittedUserDatatype::structured(
                    &[1, 1, 1, 1],
                    &[
                        offset_of!(Key, 0) as Address,
                        offset_of!(Key, 1) as Address,
                        offset_of!(Key, 2) as Address,
                        offset_of!(Key, 3) as Address,
                    ],
                    &[
                        UncommittedUserDatatype::contiguous(1, &u64::equivalent_datatype())
                            .as_ref(),
                        UncommittedUserDatatype::contiguous(1, &u64::equivalent_datatype())
                            .as_ref(),
                        UncommittedUserDatatype::contiguous(1, &u64::equivalent_datatype())
                            .as_ref(),
                        UncommittedUserDatatype::contiguous(1, &u64::equivalent_datatype())
                            .as_ref(),
                    ],
                )
                .as_ref(),
                UncommittedUserDatatype::contiguous(1, &usize::equivalent_datatype()).as_ref(),
            ],
        )
    }
}

/// Subroutine for finding the parent of a Morton key in its component representation. The trick
/// is to figure out whether the anchor of a key survives at its parent level, and notice that
/// anchors at odd indices don't survive. `parent_level_diff' refers to the difference between the
/// parent's key's level, and the maximum depth of the tree.
fn odd_index(idx: u64, parent_level_diff: u64) -> bool {
    let factor = 1 << parent_level_diff;
    (idx % factor) != 0
}

/// Find the parent of a **Morton Key**. Parents contain the key, and are at the previous level of
/// discretisation.
pub fn find_parent(key: &Key, depth: &u64) -> Key {
    // Return root if root fed in
    if (key.0 == 0) & (key.1 == 0) & (key.2 == 0) {
        match key.3 {
            0 => Key(0, 0, 0, 0),
            _ => Key(0, 0, 0, key.3 - 1),
        }
    } else {
        let level_diff = depth - key.3;
        let shift = 1 << level_diff;
        let parent_level_diff = depth - (key.3 - 1);

        let x_odd = odd_index(key.0, parent_level_diff);
        let y_odd = odd_index(key.1, parent_level_diff);
        let z_odd = odd_index(key.2, parent_level_diff);

        let mut parent = Key(key.0, key.1, key.2, key.3 - 1);

        if x_odd {
            parent.0 = key.0 - shift;
        };
        if y_odd {
            parent.1 = key.1 - shift;
        }
        if z_odd {
            parent.2 = key.2 - shift;
        }
        parent
    }
}

/// Find the siblings of a **Morton Key**. Siblings share the same parent.
pub fn find_siblings(key: &Key, depth: &u64) -> Keys {
    let parent = find_parent(key, depth);

    let mut first_child = parent;
    first_child.3 += 1;

    let mut siblings: Keys = Vec::new();

    let level_diff = depth - key.3;
    let shift = 1 << level_diff;

    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                siblings.push(Key(
                    first_child.0 + shift * i,
                    first_child.1 + shift * j,
                    first_child.2 + shift * k,
                    first_child.3,
                ));
            }
        }
    }
    siblings
}

/// Find the children of a **Morton Key**.
pub fn find_children(key: &Key, depth: &u64) -> Keys {
    let mut first_child = *key;
    first_child.3 += 1;
    find_siblings(&first_child, depth)
}

/// Encode a **Point** in a **Morton Key**.
pub fn encode_point(mut point: &mut Point, &level: &u64, &depth: &u64, &x0: &Point, &r0: &f64) {
    let mut key = Key(0, 0, 0, level);
    let mut displacement = x0;
    displacement.x = x0.x - r0;
    displacement.y = x0.y - r0;
    displacement.z = x0.z - r0;

    let side_length: f64 = (r0 * 2.) / ((1 << depth) as f64);

    key.0 = ((point.x - displacement.x) / side_length).floor() as u64;
    key.1 = ((point.y - displacement.y) / side_length).floor() as u64;
    key.2 = ((point.z - displacement.z) / side_length).floor() as u64;
    point.key = key;
}

/// Encode a vector of **Points** with their corresponding Morton keys at a given discretisation
/// in parallel.
pub fn encode_points(points: &mut [Point], level: &u64, depth: &u64, x0: &Point, r0: &f64) {
    points
        .par_iter_mut()
        .map(|p| encode_point(p, level, depth, x0, r0))
        .collect()
}

/// Find all ancestors of a **Morton Key**, excludes the key.
pub fn find_ancestors(key: &Key, depth: &u64) -> Keys {
    let root = Key(0, 0, 0, 0);
    let mut parent = find_parent(key, depth);
    let mut ancestors: Keys = vec![parent];

    while parent != root {
        parent = find_parent(&parent, depth);
        ancestors.push(parent);
    }
    ancestors
}

/// Find the finest common ancestor of two **Morton Keys**.
pub fn find_finest_common_ancestor(a: &Key, b: &Key, depth: &u64) -> Key {
    let ancestors_a: HashSet<Key> = find_ancestors(a, depth).into_iter().collect();
    let ancestors_b: HashSet<Key> = find_ancestors(b, depth).into_iter().collect();

    let intersection: HashSet<Key> = ancestors_a.intersection(&ancestors_b).copied().collect();

    intersection.into_iter().max().unwrap()
}

/// The deepest first descendent of a **Morton Key**. First descendants always share anchors.
pub fn find_deepest_first_descendent(key: &Key, depth: &u64) -> Key {
    if key.3 < *depth {
        Key(key.0, key.1, key.2, *depth)
    } else {
        *key
    }
}

/// The deepest last descendent of a **Morton Key**. At the deepest level nodes are considered to
/// have side lengths of 1.
pub fn find_deepest_last_descendent(key: &Key, depth: &u64) -> Key {
    if key.3 < *depth {
        let mut level_diff = depth - key.3;
        let mut dld = *find_children(key, depth).iter().max().unwrap();

        while level_diff > 1 {
            let tmp = dld;
            dld = *find_children(&tmp, depth).iter().max().unwrap();
            level_diff -= 1;
        }

        dld
    } else {
        *key
    }
}

/// Convert a vector of **Points**, to a Vector of **Leaves**.
pub fn keys_to_leaves(mut points: &mut [Point], ncrit: &usize) -> Leaves {
    // Sort points by Leaf key
    points.sort_by(|a, b| a.key.cmp(&b.key));

    // Find unique Leaf keys
    let mut key_indices: Vec<usize> = Vec::new();
    let mut curr = Key::default();
    let mut unique_keys: Keys = Vec::new();
    let mut key_indices: Vec<usize> = Vec::new();

    for (i, &p) in points.iter().enumerate() {
        if curr != p.key {
            curr = p.key;
            unique_keys.push(curr);
            key_indices.push(i)
        }
    }
    key_indices.push(points.len());

    let mut leaves: Leaves = Vec::new();

    for (i, &key) in unique_keys.iter().enumerate() {
        let npoints = key_indices[i + 1] - key_indices[i];

        let leaf = Leaf {
            key,
            block: Key::default(),
            npoints,
        };

        leaves.push(leaf);
    }

    leaves
}

mod tests {
    use super::*;
    use crate::data::random;
    use itertools::Itertools;

    #[test]
    fn test_find_parent() {
        let depth = 3;
        let child = Key(3, 3, 3, 3);
        let expected = Key(2, 2, 2, 2);
        let result = find_parent(&child, &depth);
        assert_eq!(result, expected);

        let depth = 3;
        let child = Key(2, 2, 2, 2);
        let expected = Key(0, 0, 0, 1);
        let result = find_parent(&child, &depth);
        assert_eq!(result, expected);

        let depth = 3;
        let child = Key(0, 0, 0, 1);
        let expected = Key(0, 0, 0, 0);
        let result = find_parent(&child, &depth);
        assert_eq!(result, expected);

        let depth = 3;
        let child = Key(0, 0, 0, 0);
        let expected = Key(0, 0, 0, 0);
        let result = find_parent(&child, &depth);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_find_siblings() {
        let depth = 3;
        let key = Key(0, 0, 0, 1);
        let level_diff = depth - key.3;
        let shift = 1 << level_diff;
        let mut expected: Keys = vec![
            Key(0, 0, 0, 1),
            Key(shift, 0, 0, 1),
            Key(0, shift, 0, 1),
            Key(0, 0, shift, 1),
            Key(0, shift, shift, 1),
            Key(shift, shift, 0, 1),
            Key(shift, 0, shift, 1),
            Key(shift, shift, shift, 1),
        ];
        expected.sort();

        let mut result = find_siblings(&key, &depth);
        result.sort();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_find_children() {
        let key = Key(0, 0, 0, 0);
        let depth = 5;
        let level_diff = depth - (key.3 + 1);
        let shift = 1 << level_diff;
        let mut expected: Keys = vec![
            Key(0, 0, 0, 1),
            Key(shift, 0, 0, 1),
            Key(0, shift, 0, 1),
            Key(0, 0, shift, 1),
            Key(0, shift, shift, 1),
            Key(shift, shift, 0, 1),
            Key(shift, 0, shift, 1),
            Key(shift, shift, shift, 1),
        ];

        expected.sort();
        let result = find_children(&key, &depth);
        assert_eq!(result, expected)
    }

    #[test]
    fn test_sorting() {
        let key = Key(0, 0, 0, 1);
        let depth = 3;
        let mut keys = find_siblings(&key, &depth);

        keys.sort();

        // Test sorting method
        let mut prev = keys[0 as usize];

        for i in 1..(keys.len() - 1) {
            let curr = keys[i as usize];
            assert!(curr > prev);
            prev = curr.clone();
        }

        // Test that level makes a difference
        let a = Key(0, 0, 0, 0);
        let b = Key(0, 0, 0, 1);
        assert!(a < b);

        // Test that children are always greater than parent
        let parent = Key(0, 0, 0, 1);
        let children = find_children(&parent, &depth);
        for child in children {
            assert!(child > parent);
            assert!(parent < child);
        }

        // Test that if given three octants, a<b<c and
        // c/∈{D(b)}, a<d<c ∀d ∈{D(b)}.
        let a = Key(0, 0, 0, 1);
        let ds = find_children(&a, &depth);
        let bs = find_siblings(&a, &depth);

        for d in ds {
            assert!(a < d);
            for &b in &bs {
                if b != a {
                    assert!(d < b);
                    assert!(a < b);
                }
            }
        }
    }

    #[test]
    fn test_encode_point() {
        let depth = 2;
        let x0 = Point {
            x: 0.5,
            y: 0.5,
            z: 0.5,
            global_idx: 0,
            key: Key::default(),
        };
        let r0 = 0.5;
        let mut point = Point::default();
        point.x = 0.0;
        point.y = 0.0;
        point.z = 0.0;
        encode_point(&mut point, &depth, &depth, &x0, &r0);
        let expected = Key(0, 0, 0, 2);
        assert_eq!(point.key, expected);
    }

    #[test]
    fn test_find_ancestors() {
        let key = Key(0, 0, 0, 2);
        let depth = 3;
        let mut expected = vec![Key(0, 0, 0, 0), Key(0, 0, 0, 1)];

        expected.sort();

        let mut result = find_ancestors(&key, &depth);
        result.sort();
        assert_eq!(expected, result);

        let key = Key(2, 0, 2, 2);
        let depth = 3;
        let mut expected = vec![Key(0, 0, 0, 0), Key(0, 0, 0, 1)];

        expected.sort();

        let mut result = find_ancestors(&key, &depth);
        result.sort();
        assert_eq!(expected, result);
    }

    #[test]
    fn test_find_dld() {
        let key = Key(0, 0, 0, 0);
        let depth = 2;
        let result = find_deepest_last_descendent(&key, &depth);
        let expected = Key(3, 3, 3, 2);
        assert_eq!(expected, result);
    }

    #[test]
    fn test_find_dfd() {
        let key = Key(1, 1, 1, 1);
        let depth = 2;
        let result = find_deepest_first_descendent(&key, &depth);
        let expected = Key(1, 1, 1, 2);
        assert_eq!(expected, result);
    }

    #[test]
    fn test_find_finest_common_ancestor() {
        let a = Key(3, 3, 3, 2);
        let b = Key(0, 0, 0, 2);
        let depth = 2;
        let expected = Key(0, 0, 0, 0);
        let result = find_finest_common_ancestor(&a, &b, &depth);
        assert_eq!(expected, result);
    }

    #[test]
    fn test_keys_to_leaves() {
        let npoints = 342;
        let ncrit = 50;
        let mut points = random(npoints);
        let level = 1;
        let depth = 1;
        let x0 = Point {
            x: 0.5,
            y: 0.5,
            z: 0.5,
            global_idx: 0,
            key: Key::default(),
        };
        let r0 = 0.5;
        encode_points(&mut points, &level, &depth, &x0, &r0);
        let unique_keys: Keys = points.iter().map(|p| p.key).unique().clone().collect();
        let leaves = keys_to_leaves(&mut points, &ncrit);

        // Test that no keys are dropped
        assert_eq!(unique_keys.len(), leaves.len());

        // Test that no points are dropped
        let mut nleaf_points = 0;
        for leaf in leaves {
            nleaf_points += leaf.npoints;
        }
        assert_eq!(npoints as usize, nleaf_points);
    }
}
