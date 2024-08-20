#!/bin/bash

rep=$1
expt=$2
it=$3
ti=$4
fi=$5

echo "Running completion (scaled) on ${rep}_${expt} with idx ${ti}"

python run/${rep}/${rep}_$expt.py --comp_scaling --it $it --ti $ti --fi $fi --ro --parts $fi