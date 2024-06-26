#!/bin/bash

expt=$1
it=$2

# python run/local/local_$expt.py --test --it $it --ti 1
# python run/local/local_$expt.py --test --it $it --ti 1 --mask --rt --parts 11
# python run/local/local_$expt.py --test --it $it --ti 1 --mask --rt --parts 30
# python run/local/local_$expt.py --test --it $it --ti 1 --mask --rt --parts 41
# python run/local/local_$expt.py --test --it $it --ti 1 --mask --ro --parts 11
# python run/local/local_$expt.py --test --it $it --ti 1 --mask --ro --parts 30
# python run/local/local_$expt.py --test --it $it --ti 1 --mask --ro --parts 41

# python run/local/local_$expt.py --test --it $it --ti 2
# python run/local/local_$expt.py --test --it $it --ti 2 --mask --rt --parts 11
# python run/local/local_$expt.py --test --it $it --ti 2 --mask --rt --parts 41 42
# python run/local/local_$expt.py --test --it $it --ti 2 --mask --rt --parts 45
# python run/local/local_$expt.py --test --it $it --ti 2 --mask --ro --parts 11
# python run/local/local_$expt.py --test --it $it --ti 2 --mask --ro --parts 41 42
# python run/local/local_$expt.py --test --it $it --ti 2 --mask --ro --parts 45

# python run/local/local_$expt.py --test --it $it --ti 3
# python run/local/local_$expt.py --test --it $it --ti 3 --mask --rt --parts 2
# python run/local/local_$expt.py --test --it $it --ti 3 --mask --rt --parts 3
# python run/local/local_$expt.py --test --it $it --ti 3 --mask --rt --parts 11
# python run/local/local_$expt.py --test --it $it --ti 3 --mask --rt --parts 30
# python run/local/local_$expt.py --test --it $it --ti 3 --mask --rt --parts 41
# python run/local/local_$expt.py --test --it $it --ti 3 --mask --ro --parts 2
# python run/local/local_$expt.py --test --it $it --ti 3 --mask --ro --parts 3
# python run/local/local_$expt.py --test --it $it --ti 3 --mask --ro --parts 11
# python run/local/local_$expt.py --test --it $it --ti 3 --mask --ro --parts 30
# python run/local/local_$expt.py --test --it $it --ti 3 --mask --ro --parts 41

# python run/local/local_$expt.py --test --it $it --ti 9
# python run/local/local_$expt.py --test --it $it --ti 9 --mask --rt --parts 11
# python run/local/local_$expt.py --test --it $it --ti 9 --mask --rt --parts 30
# python run/local/local_$expt.py --test --it $it --ti 9 --mask --rt --parts 41 42
# python run/local/local_$expt.py --test --it $it --ti 9 --mask --ro --parts 11
# python run/local/local_$expt.py --test --it $it --ti 9 --mask --ro --parts 30
# python run/local/local_$expt.py --test --it $it --ti 9 --mask --ro --parts 41 42

# python run/local/local_$expt.py --test --it $it --ti 10
# python run/local/local_$expt.py --test --it $it --ti 10 --mask --rt --parts 11
# python run/local/local_$expt.py --test --it $it --ti 10 --mask --rt --parts 30
# python run/local/local_$expt.py --test --it $it --ti 10 --mask --rt --parts 41 42
# python run/local/local_$expt.py --test --it $it --ti 10 --mask --ro --parts 11
# python run/local/local_$expt.py --test --it $it --ti 10 --mask --ro --parts 30
# python run/local/local_$expt.py --test --it $it --ti 10 --mask --ro --parts 41 42

# python run/local/local_$expt.py --test --it $it --ti 11
# python run/local/local_$expt.py --test --it $it --ti 12
# python run/local/local_$expt.py --test --it $it --ti 13
# python run/local/local_$expt.py --test --it $it --ti 14
# python run/local/local_$expt.py --test --it $it --ti 15
# python run/local/local_$expt.py --test --it $it --ti 16
# python run/local/local_$expt.py --test --it $it --ti 17
# python run/local/local_$expt.py --test --it $it --ti 18
# python run/local/local_$expt.py --test --it $it --ti 19
# python run/local/local_$expt.py --test --it $it --ti 20

# python run/local/local_$expt.py --test --it $it --ti 1 --mask --ro --parts 4
# python run/local/local_$expt.py --test --it $it --ti 2 --mask --ro --parts 4
# python run/local/local_$expt.py --test --it $it --ti 3 --mask --ro --parts 4
# python run/local/local_$expt.py --test --it $it --ti 9 --mask --ro --parts 4
# python run/local/local_$expt.py --test --it $it --ti 10 --mask --ro --parts 4
# python run/local/local_$expt.py --test --it $it --ti 11 --mask --ro --parts 4
# python run/local/local_$expt.py --test --it $it --ti 12 --mask --ro --parts 4
# python run/local/local_$expt.py --test --it $it --ti 13 --mask --ro --parts 4
# python run/local/local_$expt.py --test --it $it --ti 14 --mask --ro --parts 4
# python run/local/local_$expt.py --test --it $it --ti 15 --mask --ro --parts 4
# python run/local/local_$expt.py --test --it $it --ti 16 --mask --ro --parts 4
# python run/local/local_$expt.py --test --it $it --ti 17 --mask --ro --parts 4
# python run/local/local_$expt.py --test --it $it --ti 18 --mask --ro --parts 4
# python run/local/local_$expt.py --test --it $it --ti 19 --mask --ro --parts 4
# python run/local/local_$expt.py --test --it $it --ti 20 --mask --ro --parts 4
python run/local/local_$expt.py --test --it $it --ti 21 --mask --ro --parts 4
python run/local/local_$expt.py --test --it $it --ti 22 --mask --ro --parts 4
python run/local/local_$expt.py --test --it $it --ti 23 --mask --ro --parts 4
python run/local/local_$expt.py --test --it $it --ti 24 --mask --ro --parts 4
python run/local/local_$expt.py --test --it $it --ti 25 --mask --ro --parts 4
python run/local/local_$expt.py --test --it $it --ti 26 --mask --ro --parts 4
python run/local/local_$expt.py --test --it $it --ti 27 --mask --ro --parts 4
python run/local/local_$expt.py --test --it $it --ti 28 --mask --ro --parts 4
python run/local/local_$expt.py --test --it $it --ti 29 --mask --ro --parts 4
python run/local/local_$expt.py --test --it $it --ti 30 --mask --ro --parts 4
python run/local/local_$expt.py --test --it $it --ti 31 --mask --ro --parts 4