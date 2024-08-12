#!/bin/bash

rep=$1
expt=$2
it=$3
ti=$4

echo "Running inversion on ${rep}_${expt} with idx ${ti}"

python run/${rep}/${rep}_$expt.py --inv --it $it --ti $ti