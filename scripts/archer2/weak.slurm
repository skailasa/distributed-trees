#!/bin/bash

#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=e681
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --time=00:30:00
#SBATCH --nodes=16

# Restore AMD compiler env
# module restore PrgEnv-aocc

# Export PKG_CONFIG path, missing on Archer2
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/opt/cray/pe/mpich/8.0.16/ofi/gnu/9.1/lib/pkgconfig

# Export work and home paths
export WORK=/work/e681/e681/skailasa

# Define a scratch directory for the job
export TEST=${WORK}/distributed-trees/weak
export SCRATCH=${TEST}/${SLURM_JOBID}

# Create a scratch directory for this run
mkdir -p ${SCRATCH}
cd ${SCRATCH}

# Set simulation parameters
n_tasks=(1 2 4 8 16 32 64 128 256 512 1024)

export NCRIT=150
export NPOINTS=1000000
export DEPTH=10

# Create a CSV output file for analysis
export OUTPUT=${SCRATCH}/weak_scaling_${SLURM_JOBID}.csv
touch ${OUTPUT}
echo "n_processes, n_leaves, runtime, encoding_time, sorting_time" >> ${OUTPUT}

# Run jobs
for i in ${!n_tasks[@]}; do
    srun --ntasks=${n_tasks[$i]} --ntasks-per-core=1 --nodes=16 ${TEST}/weak >> ${OUTPUT}
done

