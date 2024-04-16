#!/bin/bash

expt=$1
oi=$2

nohup python -u run/local/local_${expt}.py --train --of --oi ${oi} &> tmp/local/local_${expt}-of-${oi}.out < /dev/null &
echo $! > tmp/local/local_${expt}-of-${oi}.txt