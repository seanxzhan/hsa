#!/bin/bash

rep=$1
expt=$2
it=$3

echo "Running assembly scaled on ${rep}_${expt} with anno_ids $4 $5 $6 $7"

python run/${rep}/${rep}_${expt}.py --asb_scaling_inv --it ${it} --anno_ids $4 $5 $6 $7 --part_indices $8 $9 ${10} ${11} 

