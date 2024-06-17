nohup python -u data_prep/preprocess_data_16.py &> tmp/preprocess_data_16.out < /dev/null &
echo $! > tmp/preprocess_data_16.txt

