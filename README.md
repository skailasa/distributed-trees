<h1 align='center'> Distributed Octrees in Rust </h1>

Distributed Octrees in Rust, construction inspired by [1, 2].

# Representation of Nodes

Node index coordinates are represented using bit-interleaved __Morton Keys__ [1], chosen for their spatial locality properties, in the usual format;

$$ z_1y_1x_1 ... z_ny_nx_n$$

We chose to represent Morton keys as 64 bit integers, and the entire tree __linearly__ as a vector of 64 bit integers.

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
export NPROCS = 4 # Number of MPI processes
export DEPTH = 3 # Maximum depth of octree

mpirun -n $NPROCS /path/to/tree
```

# Test

```bash
cargo test
```

## References
[1] Sundar, Hari, Rahul S. Sampath, and George Biros. "Bottom-up construction and 2: 1 balance refinement of linear octrees in parallel." SIAM Journal on Scientific Computing 30.5 (2008): 2675-2708.

[2] Lashuk, Ilya, et al. "A massively parallel adaptive fast-multipole method on heterogeneous architectures." Proceedings of the Conference on High Performance Computing Networking, Storage and Analysis. IEEE, 2009.