use std::cmp::Ordering;

type KeyType = u64;
/// 20 bits each for (x, y, z) coordinates and 4 bits for level
/// data
#[derive(Clone, Copy, Debug)]
pub struct Key(pub KeyType);
pub type Keys = Vec<Key>;

/// Implementation of $\left \lfloor {\log_2(.)} \right \rfloor $
pub fn log(&x: &u64) -> Option<i64> {

    match x {
        0 => None,
        _ => {

            let mut _x = x.clone();
            let mut r : i64 = 0;

            while _x > 1 {
                _x = _x >> 1;
                r += 1;
            }

            Some(r)
        }
    }
}


/// Extract 'x', 'y' or 'z' components of Morton key.
///
/// # Examples
///
/// ```
/// use tree::morton::{extract, Key};
///
/// let key = Key(0b0100100101111);
/// let e = extract(&key, 'y');
/// assert_eq!(e, 0b111);
/// ```
pub fn extract(key: &Key, comp: char) -> KeyType {

    let mut res = 0;

    let mask = match comp {
        'x' => 0b001001001001001001001001001001001001001001001001001001001001001,
        'y' => 0b010010010010010010010010010010010010010010010010010010010010010,
        'z' => 0b100100100100100100100100100100100100100100100100100100100100100,
        _ => panic!("Must select one of 'x', 'y' and 'z' for component extraction ")
    };

    // Remove level bits and apply mask
    let masked = mask & (key.0 >> 4);

    for n in (0..20).rev() {

        // Extract 3 bits in leading order
        let curr = match n {
            0 => masked & 0b111,
            _ => ((masked & (0b111 << 3*n))) >> (3*n)
        };

        res = match comp {
            'x' => res | (curr & 1),
            'y' => res | ((curr >> 1) & 1),
            'z' => res | ((curr >> 2) & 1),
            _ => panic!("Must select one of 'x', 'y' and 'z' for component extraction ")
        };
        // Add appropriate bit to the result
        if n > 0 {
            res = res << 1;
        };
    }
    res
}


/// Test Morton keys for equality.
///
/// ```
/// use tree::morton::{Key, equal};
///
/// let a = Key(0b1011111);
/// let b = Key(0b1011111);
/// assert_eq!(a, b);
/// ```
pub fn equal(a: &Key, b: &Key) -> Option<bool> {
    let result = a.0 == b.0;
    Some(result)
}


/// Subroutine of algorithm 12 in Sundar et. al.
fn _less_than(a: &Key, b: &Key) -> Option<bool> {

    let ax = extract(a, 'x');
    let ay = extract(a, 'y');
    let az = extract(a, 'z');

    let bx = extract(b, 'x');
    let by = extract(b, 'y');
    let bz = extract(b, 'z');

    let x = vec![ax^bx, ay^by, az^bz];
    let e: Vec<i64> = x.iter().map(|num| {log(&num).unwrap_or(-1)}).collect();
    let max = e.iter().max().unwrap();
    let argmax = e.iter().position(|elem| elem == max).unwrap();

    match argmax {
        0 => {if ax < bx {Some(true)} else {Some(false)}},
        1 => {if ay < by {Some(true)} else {Some(false)}},
        2 => {if az < bz {Some(true)} else {Some(false)}},
        _ => None,
    }
}

/// Test Morton keys for relative size using algorithm 12
/// in Sundar et. al.
///
/// # Example
/// ```
/// use tree::morton::{Key, less_than};
///
/// let a = Key(0b1010001);
/// let b = Key(0b1010010);
/// let result = less_than(&a, &b);
///
/// assert_eq!(result.unwrap(), true);
/// ```
pub fn less_than(a: &Key, b: &Key) -> Option<bool> {

    let al = a.0 & 0b1111;
    let bl = b.0 & 0b1111;

    if al == bl {
        _less_than(a, b)
    } else if al < bl {
        Some(true)
    } else {
        Some(false)
    }
}


impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        let result = less_than(self, other).unwrap();

        match result {
            true => Ordering::Less,
            false => Ordering::Greater
        }
    }
}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        let result = equal(self, other).unwrap();
        result
    }
}

impl Eq for Key {}




/// Returns the final 4 bits of a Morton key corresponding
/// to the octree level of a node.
///
/// # Examples
///
/// ```
/// use tree::morton::{Key, find_level};
///
/// let key = Key(4);
/// let level = find_level(&key); // 4
/// ```
pub fn find_level(key: &Key) -> u64 {
    return key.0 & 15
}

/// Returns the parent key of a Morton key.
///
/// # Examples
///
/// ```
/// use tree::morton::{Key, find_parent};
///
/// let key = Key(4);
/// let parent = find_parent(&key); // 3
/// ```
pub fn find_parent(key: &Key) -> Key {
    let parent_level = find_level(key)-1;
    Key(key.0 >> 7 << 4 | parent_level)
}


/// Find siblings of a given Morton key.
///
/// # Examples
///
/// ```
/// use tree::morton::{Key, find_siblings};
///
/// let key = Key(1);
/// let siblings = find_siblings(&key);
/// ```
pub fn find_siblings(key: &Key) -> Keys {

    let suffixes: Vec<KeyType> = vec![0, 1, 2, 3, 4, 5, 6, 7];

    let level = find_level(key);

    let mut siblings: Keys = vec![Key(0); 8];
    let root = key.0 >> 7 << 3;

    for (i, suffix) in suffixes.iter().enumerate() {
        let sibling = root | suffix;
        println!("sibling {}", sibling << 4 | level);
        siblings[i] = Key(sibling << 4 | level);
    }

    siblings
}


/// Find the children of a given Morton key.
///
/// Examples
///
/// ```
/// use tree::morton::{Key, find_children};
///
/// let key = Key(0);
/// let children = find_children(&key);
/// ```
pub fn find_children(key: &Key) -> Keys {

    let child_level = find_level(key)+1;

    // Find first child, and its siblings
    let first_child = Key(key.0 >> 4 << 7 | child_level);

    find_siblings(&first_child)
}


mod tests {
    use super::*;

    #[test]
    fn test_find_level() {
        let key = Key(1);
        let expected = 1;
        let result = find_level(&key);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_find_parent() {
        let child = Key(4);
        let expected = Key(3);
        let result = find_parent(&child);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_find_siblings() {
        let key = Key(1);
        let expected: Keys = vec![1, 17, 33, 49, 65, 81, 97, 113]
                                .iter()
                                .map(|x| {Key(*x)} )
                                .collect();
        let result = find_siblings(&key);
        assert_eq!(result, expected)
    }

    #[test]
    fn test_find_children() {
        let key = Key(0);
        let expected: Keys = vec![1, 17, 33, 49, 65, 81, 97, 113]
                                .iter()
                                .map(|x| {Key(*x)} )
                                .collect();

        let result = find_children(&key);
        assert_eq!(result, expected)
    }
}