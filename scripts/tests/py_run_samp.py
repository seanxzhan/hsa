import os
import subprocess

devnull = open(os.devnull, 'w')

rep = 'occflexi'
test = 'samp'; r = range(0, 20)
expt = 15; it = -1

for i in r:
    subprocess.call(
        [f"scripts/tests/{test}.sh {rep} {expt} {it} {i}"],
        shell=True)
