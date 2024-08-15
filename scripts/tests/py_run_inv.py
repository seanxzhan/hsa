import os
import subprocess

devnull = open(os.devnull, 'w')

rep = 'occflexi'
test = 'inv'; r = range(496, 508)
expt = 19; it = -1

for i in r:
    subprocess.call(
        [f"scripts/tests/{test}.sh {rep} {expt} {it} {i}"],
        shell=True)
