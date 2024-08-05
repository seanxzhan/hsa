#!/bin/bash

expt=$1

nohup python -u run/occflexi/occflexi_${expt}.py --train &> tmp/occflexi/occflexi_${expt}.out < /dev/null &
echo $! > tmp/occflexi/occflexi_${expt}.txt