<h1 align='center'> Distributed Octrees in Rust </h1>

Distributed Octrees in Rust, construction inspired by [1, 2].

# Representation of Nodes

Node index coordinates are represented using bit-interleaved __Morton Keys__ [1], chosen for their spatial locality properties. We store them in component form in a tuple struct, consisting of an absolute coordinate - known as the anchor - of the lower left corner, and their level,

```rust
// Initialise a Morton key (x, y, z, l)
let key = Key(0, 0, 0, 1)
```
This form is chosen as it's easier to implement sorting with absolute coordinates. We use the algorithm from [3].


We chose to represent Morton keys as 64 bit integers, and (sub)trees __linearly__ as vectors of Morton keys.

# Build

## Binary

```bash
cargo build
```

## Documentation
We use Katex for parsing Latex from doc strings, to build:

```bash
RUSTDOCFLAGS="--html-in-header /abs/path/to/docs-header.html" cargo doc
```
# Run

Number of MPI processes must be set to be a multiple or factor of 8. This is to retain compatibility with the distribution of all non-adaptive leaves in an octree.

```bash
export NPROCS=4 # Number of MPI processes
export DEPTH=3 # Maximum depth of octree
export NPOINTS=100000 # Number of points to distribute randomly (for testing)

mpirun -n $NPROCS /path/to/tree
```

# Test

```bash
cargo test
```

## References
[1] Sundar, Hari, Rahul S. Sampath, and George Biros. "Bottom-up construction and 2: 1 balance refinement of linear octrees in parallel." SIAM Journal on Scientific Computing 30.5 (2008): 2675-2708.

[2] Lashuk, Ilya, et al. "A massively parallel adaptive fast-multipole method on heterogeneous architectures." Proceedings of the Conference on High Performance Computing Networking, Storage and Analysis. IEEE, 2009.

[3] Chan, T. "Closest-point problems simplified on the RAM", ACM-SIAM Symposium on Discrete Algorithms (2002)