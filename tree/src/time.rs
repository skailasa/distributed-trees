use std::time::Instant;

pub fn timer<F: Fn()>(func: F) -> f64 {
    let start = Instant::now();
    func();
    start.elapsed().as_millis() as f64
}