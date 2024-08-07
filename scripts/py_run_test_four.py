import os
import subprocess

devnull = open(os.devnull, 'w')

rep = 'occflexi'
expt = 11
it = 8500

for i in range(0, 50):
    subprocess.call([f"scripts/test_four_one_shape.sh {rep} {expt} {it} {i}"],
                    shell=True)