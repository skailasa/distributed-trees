# 1) Kathleen (Intel Xeon Gold/ Skylake Architecture)

## Build Notes

### Hardware:

* 192 compute nodes (7680 cores in total)
* Each compute node consists of two 20-core Intel Xeon Gold 6248 2.5GHz processors, with 192 GB of RAM, and an Intel OmniPath interconnect.
* Minimum of 40 cores required for Kathleen jobs (1 Node).
* Each of the two login nodes is identical hardware, with the addition of 2x 1TB hardrive on each.

#### Xeon Gold Notes
* ISA: everything up to AVX-512

### Software:

* Kathleen's default C compiler is intel-2018, with intel MPI 2018/3.
* Distributed trees needs the LLVM module for access to libclang:
```bash
module load llvm
```
* Scheduler based on SGE (Son of a Grid Engine)

# 2) Archer 2

## Build notes

### Hardware


### Software

Work with AMD (aocc) C compiler

* ``` module restore PrgEnv-aocc```

Set PKG_CONFIG_PATH to point to the location of the mpich.pc file

Find the location of the file in /opt

* ```find /opt/cray/pe -name mpich.pc```

Set the path to match the compiler, MPICH version and networking library (OFI/UCX)

* ```export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/opt/cray/pe/mpich/8.0.16/ofi/gnu/9.1/lib/pkgconfig/```

* Standard Slurm scheduler


# 3) Isambard 2


