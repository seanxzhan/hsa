#!/bin/bash

expt=$1
it=$2

# python run/local/local_$expt.py --test --it $it --ti 0 --mask --ro --parts 4
# python run/local/local_$expt.py --test --it $it --ti 1 --mask --ro --parts 4
# python run/local/local_$expt.py --test --it $it --ti 2 --mask --ro --parts 4
# python run/local/local_$expt.py --test --it $it --ti 3 --mask --ro --parts 4
# python run/local/local_$expt.py --test --it $it --ti 4 --mask --ro --parts 4
# python run/local/local_$expt.py --test --it $it --ti 5 --mask --ro --parts 4
# exit 0
python run/local/local_$expt.py --test --it $it --ti 10 --mask --ro --parts 4
python run/local/local_$expt.py --test --it $it --ti 11 --mask --ro --parts 4
python run/local/local_$expt.py --test --it $it --ti 12 --mask --ro --parts 4
python run/local/local_$expt.py --test --it $it --ti 13 --mask --ro --parts 4
python run/local/local_$expt.py --test --it $it --ti 14 --mask --ro --parts 4
python run/local/local_$expt.py --test --it $it --ti 15 --mask --ro --parts 4
python run/local/local_$expt.py --test --it $it --ti 16 --mask --ro --parts 4
python run/local/local_$expt.py --test --it $it --ti 17 --mask --ro --parts 4
python run/local/local_$expt.py --test --it $it --ti 18 --mask --ro --parts 4
python run/local/local_$expt.py --test --it $it --ti 19 --mask --ro --parts 4
python run/local/local_$expt.py --test --it $it --ti 20 --mask --ro --parts 4