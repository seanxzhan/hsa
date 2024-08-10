import os
import subprocess

devnull = open(os.devnull, 'w')

rep = 'occflexi'
# expt = 10
# it = 6000 # for 10
# expt = 11
# it = 6750 # for 11
# expt = 13
# it = 5250 # for 13
expt = 14
it = 4000 # for 14

for i in range(0, 50):
    subprocess.call([f"scripts/test_four_one_shape.sh {rep} {expt} {it} {i}"],
                    shell=True)