#!/bin/bash

expt=$1

nohup python -u run/vae/vae_${expt}.py --train &> tmp/vae/vae_${expt}.out < /dev/null &
echo $! > tmp/vae/vae_${expt}.txt