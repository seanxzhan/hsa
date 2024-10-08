import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--expt', type=int, required=True)
args = parser.parse_args()

rep = 'occflexi'
test = 'recon'; r = range(0, 100)
expt = args.expt; it = -1

for i in r:
    subprocess.call(
        [f"scripts/tests/{test}.sh {rep} {expt} {it} {i}"],
        shell=True)
