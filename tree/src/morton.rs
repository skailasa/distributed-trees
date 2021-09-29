use std::fmt;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::collections::HashSet;

use rayon::prelude::*;

/// Maximum points for allocation of leaf point array
pub const MAX_POINTS: usize = 50;
pub const SENTINELf64: f64 = 999.;

type PointType = f64;
/// Cartesian physical coordinates (x, y, z) of a given point.
#[derive(Clone, Copy, Debug)]

pub struct Point(pub PointType, pub PointType, pub PointType);
pub type Points = Vec<Point>;

type KeyType = u64;
/// 20 bits each for (x, y, z) indices from anchor representation of Morton key,
/// 4 bits for level data.
#[derive(Clone, Copy, Debug)]
pub struct Key(pub KeyType, pub KeyType, pub KeyType, pub KeyType);
pub type Keys = Vec<Key>;

// Struct holding point data for a given leaf
#[derive(Clone, Debug)]
pub struct LeafPoints(pub [Point; MAX_POINTS]);

/// 20 bits each for (x, y, z) indices from anchor representation of Morton key,
/// 4 bits for level data.
#[derive(Clone, Debug)]
pub struct Leaf{
    pub key: Key,
    pub points: LeafPoints
}

impl std::fmt::Display for Leaf {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {

        let npoints = self.points.0.iter().filter(|&p| p.0 != SENTINELf64).count();

        write!(f, "key: {:?}, npoints: {}", self.key, npoints)
    }
}


pub type Leaves = Vec<Leaf>;


/// Test Morton keys for equality.
fn equal(a: &Key, b: &Key) -> Option<bool> {
    let result = (a.0 == b.0) & (a.1 == b.1) & (a.2 == b.2) & (a.3 == b.3);
    Some(result)
}



#[derive(Clone, Debug)]
pub struct Ke(pub KeyType, pub KeyType, pub KeyType, pub KeyType, pub LeafPoints);
pub type Kes = Vec<Ke>;

/// Subroutine in less than function, equivalent to comparing floor of log_2(x).
/// Adapted from T.Chan Chan, T. "Closest-point problems simplified on the RAM",
/// ACM-SIAM Symposium on Discrete Algorithms (2002).
fn most_significant_bit(x: u64, y: u64) -> bool {
    (x < y) & (x < (x^y))
}

/// Implementation of Algorithm 12 in Sundar et. al. as a subroutine for the
/// Ord trait. If a is less than b, this returns true.
fn less_than(a: &Key, b: &Key) -> Option<bool> {

    // If anchors match, the one at the coarser level has the lesser Morton id.
    let same_anchor = (a.0 == b.0) & (a.1 == b.1) & (a.2 == b.2);

    match same_anchor {
        true => {if a.3 < b.3 {Some(true)} else {Some(false)}},
        false => {
            let x = vec![a.0^b.0, a.1^b.1, a.2^b.2];

            let mut argmax = 0;

            for dim in 1..3 {
                if most_significant_bit(x[argmax as usize], x[dim as usize]) {
                    argmax = dim
                }
            }

            match argmax {
                0 => {if a.0 < b.0 {Some(true)} else {Some(false)}},
                1 => {if a.1 < b.1 {Some(true)} else {Some(false)}},
                2 => {if a.2 < b.2 {Some(true)} else {Some(false)}},
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
            false => {
                match less {
                    true => Some(Ordering::Less),
                    false => Some(Ordering::Greater)
                }
            }
        }
    }
}

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        let result = equal(self, other).unwrap();
        result
    }
}

impl Eq for Key {}

/// A unique hash for a Morton key is simply it's four components.
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
        let eq = self.eq(&other);

        match eq {
            true => Some(Ordering::Equal),
            false => {
                match less {
                    true => Some(Ordering::Less),
                    false => Some(Ordering::Greater)
                }
            }
        }
    }
}

impl PartialEq for Leaf {
    fn eq(&self, other: &Self) -> bool {
        let result = equal(&self.key, &other.key).unwrap();
        result
    }
}

impl Eq for Leaf {}

/// A unique hash for a Morton key is simply it's four components.
impl Hash for Leaf {

    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.0.hash(state);
        self.key.1.hash(state);
        self.key.2.hash(state);
        self.key.3.hash(state);
    }
}


/// Subroutine for finding the parent of a Morton key in its component
/// representation.
///
/// The trick is to figure out whether the anchor of a key
/// survives at its parent level, and notice that anchors at odd indices
/// don't survive. `parent_level_diff' refers to the difference between
/// the parent's key's level, and the maximum depth of the tree.
fn odd_index(idx: u64, parent_level_diff: u64) -> bool {
    let factor = 1 << parent_level_diff;

    if (idx % factor) == 0 {
        false
    } else {
        true
    }
}


/// Find the parent of a Morton key.
pub fn find_parent(key: &Key, depth: &u64) -> Key {

    // Return root if root fed in
    if (key.0 == 0) & (key.1 == 0) & (key.2 == 0) {
        match key.3 {
            0 => Key(0, 0, 0, 0),
            _ => Key(0, 0, 0, key.3-1)
        }
    } else {

        let level_diff = depth-key.3;
        let shift = 1 << level_diff;
        let parent_level_diff = depth-(key.3-1);

        let x_odd = odd_index(key.0, parent_level_diff);
        let y_odd = odd_index(key.1, parent_level_diff);
        let z_odd = odd_index(key.2, parent_level_diff);

        let mut parent = Key(key.0, key.1, key.2, key.3-1);

        if x_odd {
            parent.0 = key.0-shift;
        };
        if y_odd {
            parent.1 = key.1-shift;
        }
        if z_odd {
            parent.2 = key.2-shift;
        }
        parent
    }
}


/// Find the siblings of a Morton key. Siblings share
/// the same parent.
pub fn find_siblings(key: &Key, depth: &u64) -> Keys {
    let parent = find_parent(key, depth);

    let mut first_child = parent.clone();
    first_child.3 += 1;

    let mut siblings: Keys = Vec::new();

    let level_diff = depth-key.3;
    let shift = 1 << level_diff;

    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                siblings.push(
                    Key(
                        first_child.0+shift*i,
                        first_child.1+shift*j,
                        first_child.2+shift*k,
                        first_child.3
                    )
                );
            }
        }
    }
    siblings
}


/// Find the children of a Morton key.
pub fn find_children(key: &Key, depth: &u64) -> Keys {

    let mut first_child = key.clone();
    first_child.3 += 1;
    find_siblings(&first_child, depth)
}


/// Encode a Cartesian coordinate in a Morton key.
pub fn encode_point(
    &point: &Point, &level: &u64, &depth: &u64, &x0: &Point, &r0: &f64
) -> Key {

    let mut key = Key(0, 0, 0, level);
    let displacement = Point(x0.0 - r0, x0.1 - r0, x0.2 - r0);
    let side_length: f64 = (r0*2.) / ((1 << depth) as f64);

    key.0 = ((point.0 - displacement.0) / side_length).floor() as u64;
    key.1 = ((point.1 - displacement.1) / side_length).floor() as u64;
    key.2 = ((point.2 - displacement.2) / side_length).floor() as u64;
    key
}


/// Encode a vector of physical point coordinates into their corresponding
/// Morton keys, in parallel.
pub fn encode_points(
    points: &Points, level: &u64, depth: &u64, x0: &Point, r0: &f64
) -> Keys {

    let keys = points.par_iter()
                     .map(|p| encode_point(&p, level, depth, x0, r0))
                     .collect();

    return keys
}


/// Find ancestors of a Morton key.
pub fn find_ancestors(key: &Key, depth: &u64) -> Keys {

    let root = Key(0, 0, 0, 0);
    let mut parent = find_parent(key, depth);
    let mut ancestors : Keys = vec![parent];

    while parent != root {
        parent = find_parent(&parent, depth);
        ancestors.push(parent);
    }
    ancestors
}

/// Find the finest common ancestor of two Morton keys.
pub fn find_finest_common_ancestor(a: &Key, b: &Key, depth: &u64) -> Key {

    let ancestors_a: HashSet<Key> = find_ancestors(a, depth).into_iter()
                                                            .collect();
    let ancestors_b: HashSet<Key> = find_ancestors(b, depth).into_iter()
                                                            .collect();

    let intersection: HashSet<Key> = ancestors_a.intersection(&ancestors_b)
                                                 .copied()
                                                 .collect();

    let intersection: Vec<Key> = intersection.into_iter()
                                             .collect();

    intersection.into_iter()
                .max()
                .unwrap()
}


/// The deepest first descendent of a Morton key.
/// First descendents always share anchors.
pub fn find_deepest_first_descendent(
    key: &Key, depth: &u64
) -> Key {
    if key.3 < *depth {
        Key(key.0, key.1, key.2, depth.clone())
    } else {
        key.clone()
    }
}


/// The deepest last descendent of a Morton key.
/// At the deepest level, Keys are considered to have
/// have side lengths of 1.
pub fn find_deepest_last_descendent(
    key: &Key, depth: &u64
) -> Key {

    if key.3 < *depth {

        let mut level_diff = depth-key.3;
        let mut dld = find_children(key, depth).iter().max().unwrap().clone();

        while level_diff > 1 {
            let tmp = dld.clone();
            dld = find_children(&tmp, depth).iter().max().unwrap().clone();
            level_diff -= 1;
        }

        dld
    } else {
        key.clone()
    }
}

pub fn find_descendents(key: &Key, level: &u64, depth: &u64) -> Keys {

    let mut descendents: Keys = vec![key.clone()];
    let mut level_diff = depth-level;

    while level_diff > 0 {

        let mut aux: Keys = Vec::new();

        for d in descendents {
            let mut children = find_children(&d, depth);
            aux.append(&mut children);
        }

        descendents = aux;
        level_diff -= 1;
    }
    descendents
}

mod tests {
    use super::*;

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
        let level_diff = depth-key.3;
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
        let level_diff = depth-(key.3+1);
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

        for i in 1..(keys.len()-1) {
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
        };

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
    fn test_find_ancestors() {
        let key = Key(0, 0, 0, 2);
        let depth = 3;
        let mut expected = vec![
            Key(0, 0, 0, 0), Key(0, 0, 0, 1)
        ];

        expected.sort();

        let mut result = find_ancestors(&key, &depth);
        result.sort();
        assert_eq!(expected, result);

        let key = Key(2, 0, 2, 2);
        let depth = 3;
        let mut expected = vec![
            Key(0, 0, 0, 0), Key(0, 0, 0, 1)
        ];

        expected.sort();

        let mut result = find_ancestors(&key, &depth);
        result.sort();
        assert_eq!(expected, result);
    }
}