#!/bin/bash

ARGS=(2 3 4 5 6)
echo "Running local experiments: ${ARGS[@]}"

for arg in "${ARGS[@]}"
do
    echo "Running local_${arg}.py"
    nohup python -u run/local/local_${arg}.py --train &> tmp/local/local_${arg}.out < /dev/null &
    echo $! > tmp/local/local_${arg}.txt
    wait
done 

echo "All local expt runs have completed."