#!/bin/bash

expt=$1
it=$2

python run/local/local_$expt.py --test --it $it --ti 1
python run/local/local_$expt.py --test --it $it --ti 1 --mask --rt --parts 11
python run/local/local_$expt.py --test --it $it --ti 1 --mask --rt --parts 30
python run/local/local_$expt.py --test --it $it --ti 1 --mask --rt --parts 41
python run/local/local_$expt.py --test --it $it --ti 1 --mask --ro --parts 11
python run/local/local_$expt.py --test --it $it --ti 1 --mask --ro --parts 30
python run/local/local_$expt.py --test --it $it --ti 1 --mask --ro --parts 41

python run/local/local_$expt.py --test --it $it --ti 2
python run/local/local_$expt.py --test --it $it --ti 2 --mask --rt --parts 11
python run/local/local_$expt.py --test --it $it --ti 2 --mask --rt --parts 41 42
python run/local/local_$expt.py --test --it $it --ti 2 --mask --rt --parts 45
python run/local/local_$expt.py --test --it $it --ti 2 --mask --ro --parts 11
python run/local/local_$expt.py --test --it $it --ti 2 --mask --ro --parts 41 42
python run/local/local_$expt.py --test --it $it --ti 2 --mask --ro --parts 45

python run/local/local_$expt.py --test --it $it --ti 3
python run/local/local_$expt.py --test --it $it --ti 3 --mask --rt --parts 2
python run/local/local_$expt.py --test --it $it --ti 3 --mask --rt --parts 3
python run/local/local_$expt.py --test --it $it --ti 3 --mask --rt --parts 11
python run/local/local_$expt.py --test --it $it --ti 3 --mask --rt --parts 30
python run/local/local_$expt.py --test --it $it --ti 3 --mask --rt --parts 41
python run/local/local_$expt.py --test --it $it --ti 3 --mask --ro --parts 2
python run/local/local_$expt.py --test --it $it --ti 3 --mask --ro --parts 3
python run/local/local_$expt.py --test --it $it --ti 3 --mask --ro --parts 11
python run/local/local_$expt.py --test --it $it --ti 3 --mask --ro --parts 30
python run/local/local_$expt.py --test --it $it --ti 3 --mask --ro --parts 41