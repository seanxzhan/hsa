import os
import subprocess

devnull = open(os.devnull, 'w')

# for i in range(50, 100):
for i in [39, 86, 43, 41]:
    subprocess.call([f"scripts/test_local_four_one_shape.sh 34 3000 {i}"],
                    shell=True)