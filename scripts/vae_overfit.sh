#!/bin/bash

expt=$1
oi=$2

nohup python -u run/vae/vae_${expt}.py --train --of --oi ${oi} &> tmp/vae/vae_${expt}-of-${oi}.out < /dev/null &
echo $! > tmp/vae/vae_${expt}-of-${oi}.txt