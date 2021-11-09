# 1) Kathleen (Intel Xeon Gold/ Skylake Architecture)

## Build Notes

### Kathleen hardware:

* 192 compute nodes (7680 cores in total)
* Each compute node consists of two 20-core Intel Xeon Gold 6248 2.5GHz processors, with 192 GB of RAM, and an Intel OmniPath interconnect.
* Minimum of 40 cores required for Kathleen jobs (1 Node).
* Each of the two login nodes is identical hardware, with the addition of 2x 1TB hardrive on each.

#### Xeon Gold Notes
* ISA: everything up to AVX-512

### Kathleen software:

* Kathleen's default C compiler is intel-2018, with intel MPI 2018/3.
* Distributed trees needs the LLVM module for access to libclang:

```bash
module load llvm
```

# 2) Archer2


# 3) Isambard

