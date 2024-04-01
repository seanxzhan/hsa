#!/bin/bash

expt=$1

nohup python -u run/local/local_${expt}.py --train &> tmp/local/local_${expt}.out < /dev/null &
echo $! > tmp/local/local_${expt}.txt