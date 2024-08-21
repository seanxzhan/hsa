import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--expt', type=int, required=True)
args = parser.parse_args()

rep = 'occflexi'
test = 'inv'; r = range(3000, 3272)
expt = args.expt; it = -1

for i in r:
    subprocess.call(
        [f"scripts/tests/{test}.sh {rep} {expt} {it} {i}"],
        shell=True)
