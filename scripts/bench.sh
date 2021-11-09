#!/bin/bash -l

# Request time (hours:minutes:seconds)
#$ -l h_rt=0:30:0

# Request memory per core
#$ -l mem=2G

# Set name
#$ -N MPI Test

# Request cores
# -pe mpi 80

# Set working directory
#$ -wd /home/ucapska/Scratch/output

# Run the job
gerun $HOME/src/distributed-trees/build/debug/tree
