mode=$1
expt=$2

nohup python -u scripts/tests/py_run_${mode}.py --expt ${expt} &> tmp/run_${mode}.out < /dev/null &
echo $! > tmp/run_${mode}.txt