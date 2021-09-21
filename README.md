<h1 align='center'> Distributed Octrees in Rust </h1>

Distributed Octrees in Rust, construction inspired by [1, 2].

# Representation of Nodes

# Build

## Documentation

We use Katex for parsing Latex from doc strings, to build:

```rust
RUSTDOCFLAGS="--html-in-header src/docs-header.html" cargo doc
```



Node index coordinates are represented using bit-interleaved __Morton Keys__ [1], chosen for their spatial locality properties.

We chose to represent Morton keys as 64 bit integers, and the entire tree __linearly__ as a vector of 64 bit integers.

## References
[1] Sundar, Hari, Rahul S. Sampath, and George Biros. "Bottom-up construction and 2: 1 balance refinement of linear octrees in parallel." SIAM Journal on Scientific Computing 30.5 (2008): 2675-2708.

[2] Lashuk, Ilya, et al. "A massively parallel adaptive fast-multipole method on heterogeneous architectures." Proceedings of the Conference on High Performance Computing Networking, Storage and Analysis. IEEE, 2009.