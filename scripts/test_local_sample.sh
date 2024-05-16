#!/bin/bash

expt=$1
it=$2
ref_model=$3
sample_idx=$4

python run/local/local_${expt}.py --samp --it ${it} --ti $ref_model --si $sample_idx

