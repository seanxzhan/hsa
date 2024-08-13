import os
import subprocess

devnull = open(os.devnull, 'w')

rep = 'occflexi'
test = 'recon'; r = range(0, 50)
expt = 15; it = -1

for i in r:
    subprocess.call(
        [f"scripts/tests/{test}.sh {rep} {expt} {it} {i}"],
        shell=True)
