<h1 align='center'> Distributed Octrees in Rust </h1>

Distributed Octrees in Rust, construction inspired by [1, 2].

# Representation of Nodes

Node index coordinates are represented using __Morton Keys__ [1], chosen for their spatial locality properties. We store them in component form in a tuple struct, consisting of an absolute coordinate - known as the anchor - of the lower left corner, and their level,

```rust
// Initialise a Morton key (x, y, z, l)
let key = Key(0, 0, 0, 1)
```
This form is chosen as it's easier to implement sorting with absolute coordinates. We use the algorithm from [3].

# Algorithm

Tree construction consists of a __building__ and a __balancing__ phase.

We initially build an unbalanced tree from particle coordinate data distributed across a cluster. This unbalanced tree is then 2:1 balanced.

## Building Phase

[X] indicates whether this functionality has been completed.

1. Apply Morton encoding to distributed point data at each processor. [X]

2. Apply a Parallel (Bitonic) sort to the Morton keys, such that the rank 0 process has the least keys, and the rank NPROC-1 process has the greatest keys. [X]

3. Remove the duplicates and overlaps for Morton keys on each process if they exist. [X]

4. Complete the region between the least and greatest Morton key at each process to find the coarsest possible nodes that can occupy the domain that they specify. This is algorithm 3 in [1]. The coarsest keys at each processors are now called the 'seeds'. [X]

5. Complete the region between the seeds across all processors. The elements of this final complete linear tree are called 'blocks'. [X]

6. Compute load of each block and repartition them such that each processor has a similar load. Load is estimated by computing the number of leaf octants that they could hold. [X]

7. Partition the blocks to satisfy the NCRIT value specified by the user. [X]

## Balancing Phase

# Build

## Binary

```bash
source .env && cd tree && cargo build --release
```

## Documentation
We use Katex for parsing Latex from doc strings, to build:

```bash
RUSTDOCFLAGS="--html-in-header /abs/path/to/docs-header.html" cargo doc
```

## Settings
Set tree parameters in `.env` file

```bash
# .env file
export NPROCS=4 # Number of MPI processes
export DEPTH=3 # Maximum depth of octree
export NPOINTS=100000 # Number of points to distribute randomly (for testing)
export NCRIT=100 # Maximum number of points per leaf node
export CRAY=0 # '0' if not on cray system, '1' otherwise.
```

```bash
source .env # Source before using library.
```

# Test

Run serial tests for sequential functions in tree library:

```bash
cd tree && cargo test
```

Run tests for public MPI functions:

```bash
cd parallel_tests && cargo mpirun --bin parallel_tests
```

These also act as examples for usage of the library.

# Scaling

Build binaries for Strong and Weak scaling tests, intended for cluster deployment.

```bash
cd scaling_tests && cargo build --release
```

## References
[1] Sundar, Hari, Rahul S. Sampath, and George Biros. "Bottom-up construction and 2: 1 balance refinement of linear octrees in parallel." SIAM Journal on Scientific Computing 30.5 (2008): 2675-2708.

[2] Lashuk, Ilya, et al. "A massively parallel adaptive fast-multipole method on heterogeneous architectures." Proceedings of the Conference on High Performance Computing Networking, Storage and Analysis. IEEE, 2009.

[3] Chan, T. "Closest-point problems simplified on the RAM", ACM-SIAM Symposium on Discrete Algorithms (2002)
