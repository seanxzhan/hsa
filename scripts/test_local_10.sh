#!/bin/bash

expt=$1
it=$2

python run/local/local_$expt.py --test --it ${it} --ti 1
python run/local/local_$expt.py --test --it ${it} --ti 2
python run/local/local_$expt.py --test --it ${it} --ti 4
python run/local/local_$expt.py --test --it ${it} --ti 6
python run/local/local_$expt.py --test --it ${it} --ti 1 --mask --rt --parts 6
python run/local/local_$expt.py --test --it ${it} --ti 1 --mask --rt --parts 13
python run/local/local_$expt.py --test --it ${it} --ti 1 --mask --rt --parts 18
python run/local/local_$expt.py --test --it ${it} --ti 1 --mask --ro --parts 6
python run/local/local_$expt.py --test --it ${it} --ti 1 --mask --ro --parts 13
python run/local/local_$expt.py --test --it ${it} --ti 1 --mask --ro --parts 18
python run/local/local_$expt.py --test --it ${it} --ti 2 --mask --rt --parts 6
python run/local/local_$expt.py --test --it ${it} --ti 2 --mask --rt --parts 18 19
python run/local/local_$expt.py --test --it ${it} --ti 2 --mask --rt --parts 20
python run/local/local_$expt.py --test --it ${it} --ti 4 --mask --rt --parts 6 7
python run/local/local_$expt.py --test --it ${it} --ti 4 --mask --rt --parts 11
python run/local/local_$expt.py --test --it ${it} --ti 4 --mask --rt --parts 13
python run/local/local_$expt.py --test --it ${it} --ti 6 --mask --rt --parts 6
python run/local/local_$expt.py --test --it ${it} --ti 6 --mask --rt --parts 12 13
python run/local/local_$expt.py --test --it ${it} --ti 6 --mask --rt --parts 18
