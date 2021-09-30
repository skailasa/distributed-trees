use rand::{Rng};

use crate::morton::{Point, PointsVec};

/// Generate random distribution of PointsVec in range [0, 1),
/// for testing.
pub fn random(npoints: u64) -> PointsVec {
    let mut range = rand::thread_rng();

    let mut points : PointsVec = Vec::new();

    for _ in 0..npoints {
        let x : f64 = range.gen();
        let y : f64 = range.gen();
        let z : f64 = range.gen();
        points.push(Point(x, y, z));
    }

   points
}
