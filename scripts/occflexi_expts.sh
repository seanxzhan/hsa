#!/bin/bash

ARGS=(15 16)
echo "Running occflexi experiments: ${ARGS[@]}"

for arg in "${ARGS[@]}"
do
    echo "Running occflexi_${arg}.py"
    nohup python -u run/occflexi/occflexi_${arg}.py --train &> tmp/occflexi/occflexi_${arg}.out < /dev/null &
    echo $! > tmp/occflexi/occflexi_${arg}.txt
    wait
done 

echo "All occflexi expt runs have completed."