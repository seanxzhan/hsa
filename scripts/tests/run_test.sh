mode=$1

nohup python -u scripts/tests/py_run_${mode}.py &> tmp/run_${mode}.out < /dev/null &
echo $! > tmp/run_${mode}.txt