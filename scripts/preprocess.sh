nohup python -u data_prep/preprocess_data_2.py &> tmp/preprocess_data_2.out < /dev/null &
echo $! > tmp/preprocess_data_2.txt

