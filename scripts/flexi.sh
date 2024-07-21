#!/bin/bash

expt=$1

nohup python -u run/flexi/flexi_${expt}.py --train &> tmp/flexi/flexi_${expt}.out < /dev/null &
echo $! > tmp/flexi/flexi_${expt}.txt