//! Distributed Octrees in Rust
//!
//! Octrees parallelized using Rust and MPI, for scientific computing.

/// Perform operations on Morton keys.
pub mod morton;

/// Create octrees in parallel.
pub mod tree;

/// Data manipulation and generation tools.
pub mod data;

pub mod sort;