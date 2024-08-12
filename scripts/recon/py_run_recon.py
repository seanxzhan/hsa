import os
import subprocess

devnull = open(os.devnull, 'w')

rep = 'occflexi'
# expt = 10; it = 6000
# expt = 11; it = 6750
# expt = 13; it = 5250
# expt = 14; it = 4000
expt = 15; it = -1

for i in range(0, 50):
    subprocess.call(
        [f"scripts/recon/recon_four_one_shape.sh {rep} {expt} {it} {i}"],
        shell=True)
