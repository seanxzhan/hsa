#!/bin/bash

ARGS=(15 16)
echo "Running flexi experiments: ${ARGS[@]}"

for arg in "${ARGS[@]}"
do
    echo "Running flexi_${arg}.py"
    nohup python -u run/flexi/flexi_${arg}.py --train &> tmp/flexi/flexi_${arg}.out < /dev/null &
    echo $! > tmp/flexi/flexi_${arg}.txt
    wait
done 

echo "All flexi expt runs have completed."