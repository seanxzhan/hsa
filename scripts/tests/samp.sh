#!/bin/bash

rep=$1
expt=$2
it=$3
ti=$4

echo "Running sampling on ${rep}_${expt} with idx ${ti}"

python run/${rep}/${rep}_$expt.py --samp --it $it --ti $ti