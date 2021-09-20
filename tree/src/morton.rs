use std::cmp::Ordering;

type KeyType = u64;
/// 20 bits each for (x, y, z) coordinates and 4 bits for level
/// data
#[derive(Clone, Copy, Debug)]
pub struct Key(pub KeyType);
pub type Keys = Vec<Key>;


/// Extract components of Morton key
pub fn extract(a: &Key, c: char) -> KeyType {

    let res = 1 << 20;

    let mask = match c {
        'x' => 0b001001001001001001001001001001001001001001001001,
        'y' => 0b010010010010010010010010010010010010010010010010,
        'z' => 0b100100100100100100100100100100100100100100100100,
        _ => panic!("Must select one of 'x', 'y' and 'z' for component extraction ")
    };

    let mut masked = mask & a.0;

    for n in 19..=0 {

        // Extract last 3 bits
        let curr = masked & 0b111;

        // Add appropriate bit to the result
        match c {
            'x' => res | ((curr & 1) << n),
            'y' => res | (((curr >> 1) & 1) << n),
            'z' => res | (((curr >> 2) & 1) << n),
            _ => panic!("Must select one of 'x', 'y' and 'z' for component extraction ")
        };

        // Shift down mask
        masked = masked >> 3;

    }

    // Return 20 bits of component
    res & ((1 << 20) - 1)
}


/// Test Morton keys for equality
pub fn equal(a: &Key, b: &Key) -> Option<bool> {
    let result = a.0 == b.0;
    Some(result)
}


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

/// Test Morton keys for relative size.
pub fn less_than(a: &Key, b: &Key) -> Option<bool> {

    let al = a.0 & 15;
    let bl = a.0 & 15;

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


/// \floor{log_2(.)}
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