nohup python -u scripts/tests/py_run_test.py &> tmp/run_test.out < /dev/null &
echo $! > tmp/run_test.txt