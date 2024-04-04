nohup python -u data_prep/preprocess_data_3.py &> tmp/preprocess_data_3.out < /dev/null &
echo $! > tmp/preprocess_data_3.txt

