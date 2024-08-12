import os
import subprocess

devnull = open(os.devnull, 'w')

rep = 'occflexi'
test = 'inv'
expt = 15; it = -1

for i in range(496, 508):
    subprocess.call(
        [f"scripts/tests/{test}.sh {rep} {expt} {it} {i}"],
        shell=True)
