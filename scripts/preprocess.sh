nohup python -u data_prep/preprocess_data_xform_part_class.py &> tmp/preprocess_data_xform_part_class.out < /dev/null &
echo $! > tmp/preprocess_data_xform_part_class.txt

