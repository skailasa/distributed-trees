//! Distributed Octrees in Rust
//!
//! Octrees parallelized using Rust and MPI, for scientific computing.
//!
//! # References
//! [1] Sundar, Hari, Rahul S. Sampath, and George Biros. "Bottom-up construction and 2: 1
//! balance refinement of linear octrees in parallel." SIAM Journal on Scientific Computing 30.5
//! (2008): 2675-2708.
//!
//! [2] Lashuk, Ilya, et al. "A massively parallel adaptive fast-multipole method on heterogeneous
//! architectures." Proceedings of the Conference on High Performance Computing Networking, Storage
//! and Analysis. IEEE (2009).
//!
//! [3] Chan, T. "Closest-point problems simplified on the RAM", ACM-SIAM Symposium on Discrete
//! Algorithms (2002)

/// Perform operations on Morton keys.
pub mod morton;

/// Create octrees in parallel.
pub mod tree;

/// Data manipulation and generation tools.
pub mod data;
