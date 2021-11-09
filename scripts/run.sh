#!/bin/bash -l

export NPROCS=80
export DEPTH=8
export NPOINTS=100000
export NCRIT=100

qsub bench.sh
