nohup python -u data_prep/preprocess_data_20.py &> tmp/preprocess_data_20.out < /dev/null &
echo $! > tmp/preprocess_data_20.txt

