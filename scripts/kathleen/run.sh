#!/bin/bash -l

export NPROCS=$NPROC
export DEPTH=$DEPTH
export NPOINTS=$NPOINTS
export NCRIT=$NCRIT


qsub -V bench.sh
