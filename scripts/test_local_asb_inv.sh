#!/bin/bash

expt=$1
it=$2

python run/local/local_${expt}.py --asb --inv --it ${it} --part_indices $3 $4 $5 $6

