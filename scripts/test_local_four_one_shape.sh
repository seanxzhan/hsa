#!/bin/bash

expt=$1
it=$2
ti=$3

echo "Running local_${expt} on idx ${ti}"

python run/local/local_$expt.py --test --it $it --ti $ti
python run/local/local_$expt.py --test --it $it --ti $ti --mask --ro --parts 0
python run/local/local_$expt.py --test --it $it --ti $ti --mask --ro --parts 1
python run/local/local_$expt.py --test --it $it --ti $ti --mask --ro --parts 2
python run/local/local_$expt.py --test --it $it --ti $ti --mask --ro --parts 3

