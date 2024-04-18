nohup python -u data_prep/preprocess_data_10.py &> tmp/preprocess_data_10.out < /dev/null &
echo $! > tmp/preprocess_data_10.txt

