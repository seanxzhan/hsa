nohup python -u data_prep/preprocess_data_0.py &> tmp/preprocess_data_0.out < /dev/null &
echo $! > tmp/preprocess_data_0.txt

