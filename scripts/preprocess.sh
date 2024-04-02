nohup python -u data_prep/preprocess_data_1.py &> tmp/preprocess_data_1.out < /dev/null &
echo $! > tmp/preprocess_data_1.txt

