use rand::Rng;

use crate::morton::{Point, Points};

/// Generate random distribution of PointsVec in range [0, 1),
/// for testing.
pub fn random(npoints: u64) -> Points {
    let mut range = rand::thread_rng();

    let mut points: Points = Vec::new();

    for _ in 0..npoints {
        let x: f64 = range.gen();
        let y: f64 = range.gen();
        let z: f64 = range.gen();
        let mut p = Point::default();
        p.x = x;
        p.y = y;
        p.z = z;
        points.push(p);
    }

    points
}
