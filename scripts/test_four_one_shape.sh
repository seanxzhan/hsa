#!/bin/bash

rep=$1
expt=$2
it=$3
ti=$4


echo "Running ${rep}_${expt} on idx ${ti}"

python run/${rep}/${rep}_$expt.py --test --it $it --ti $ti
python run/${rep}/${rep}_$expt.py --test --it $it --ti $ti --mask --ro --parts 0
python run/${rep}/${rep}_$expt.py --test --it $it --ti $ti --mask --ro --parts 1
python run/${rep}/${rep}_$expt.py --test --it $it --ti $ti --mask --ro --parts 2
python run/${rep}/${rep}_$expt.py --test --it $it --ti $ti --mask --ro --parts 3

