#!/bin/bash

expt=$1
it=$2
ti=$3

python run/local/local_$expt.py --test --it $it --ti $ti
python run/local/local_$expt.py --test --it $it --ti $ti --mask --ro --parts 0
python run/local/local_$expt.py --test --it $it --ti $ti --mask --ro --parts 1
python run/local/local_$expt.py --test --it $it --ti $ti --mask --ro --parts 2
python run/local/local_$expt.py --test --it $it --ti $ti --mask --ro --parts 3

