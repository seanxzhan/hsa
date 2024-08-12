#!/bin/bash

expt=$1
it=$2

python run/local/local_${expt}.py --asb --it ${it} --anno_ids $3 $4 $5 $6

