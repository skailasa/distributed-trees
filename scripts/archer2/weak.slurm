#!/bin/bash

#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=e681
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --time=00:05:00

# Restore AMD compiler env
# module restore PrgEnv-aocc

# Export PKG_CONFIG path, missing on Archer2
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/opt/cray/pe/mpich/8.0.16/ofi/gnu/9.1/lib/pkgconfig

# Export work and home paths
export WORK=/work/e681/e681/skailasa

# Define a scratch directory for the job
export TEST=${WORK}/distributed-trees/
export SCRATCH=${TEST}/${SLURM_JOBID}

# Create a scratch directory for this run
mkdir -p ${SCRATCH}
cd ${SCRATCH}

# Set simulation parameters
n_tasks_per_node=(1 2 4 8 16 32)
n_nodes=(1 1 1 1 1 1)

export NCRIT=100
export NPOINTS=1000000
export DEPTH=10

# Create a CSV output file for analysis
export OUTPUT=${SCRATCH}/weak_scaling_${SLURM_JOBID}.csv
touch ${OUTPUT}
echo "World Size, Total Leaves, Runtime" >> ${OUTPUT}

# Run jobs
for i in ${!n_tasks_per_node[@]}; do
    srun --ntasks-per-node=${n_tasks_per_node[$i]} --nodes=${n_nodes[$i]} ${TEST}/tree >> ${OUTPUT}
done

