#!/bin/bash

nohup python -u run/local/local_0.py --train &> tmp/local/local_0.out < /dev/null &
echo $! > tmp/local/local_0.txt