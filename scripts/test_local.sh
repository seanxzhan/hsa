#!/bin/bash

expt=$1
it=$2

# python run/local/local_$expt.py --test --it $it --ti 0
# python run/local/local_$expt.py --test --it $it --ti 0 --mask --rt --parts 0
# python run/local/local_$expt.py --test --it $it --ti 0 --mask --rt --parts 1
# python run/local/local_$expt.py --test --it $it --ti 0 --mask --rt --parts 2
# python run/local/local_$expt.py --test --it $it --ti 0 --mask --ro --parts 0
# python run/local/local_$expt.py --test --it $it --ti 0 --mask --ro --parts 1
# python run/local/local_$expt.py --test --it $it --ti 0 --mask --ro --parts 2

# python run/local/local_$expt.py --test --it $it --ti 1
# python run/local/local_$expt.py --test --it $it --ti 1 --mask --rt --parts 0
# python run/local/local_$expt.py --test --it $it --ti 1 --mask --rt --parts 1
# python run/local/local_$expt.py --test --it $it --ti 1 --mask --rt --parts 2
# python run/local/local_$expt.py --test --it $it --ti 1 --mask --ro --parts 0
# python run/local/local_$expt.py --test --it $it --ti 1 --mask --ro --parts 1
# python run/local/local_$expt.py --test --it $it --ti 1 --mask --ro --parts 2

# python run/local/local_$expt.py --test --it $it --ti 2
# python run/local/local_$expt.py --test --it $it --ti 2 --mask --rt --parts 0
# python run/local/local_$expt.py --test --it $it --ti 2 --mask --rt --parts 1
# python run/local/local_$expt.py --test --it $it --ti 2 --mask --rt --parts 2
# python run/local/local_$expt.py --test --it $it --ti 2 --mask --ro --parts 0
# python run/local/local_$expt.py --test --it $it --ti 2 --mask --ro --parts 1
# python run/local/local_$expt.py --test --it $it --ti 2 --mask --ro --parts 2

python run/local/local_$expt.py --test --it $it --ti 3
python run/local/local_$expt.py --test --it $it --ti 4
python run/local/local_$expt.py --test --it $it --ti 5
python run/local/local_$expt.py --test --it $it --ti 6
python run/local/local_$expt.py --test --it $it --ti 7 
python run/local/local_$expt.py --test --it $it --ti 8