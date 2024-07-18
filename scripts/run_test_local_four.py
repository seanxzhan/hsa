import os
import subprocess

devnull = open(os.devnull, 'w')

for i in range(0, 50):
# for i in [39, 86, 43, 41]:
    subprocess.call([f"scripts/test_local_four_one_shape.sh 100 3000 {i}"],
                    shell=True)