import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--expt', type=int, required=True)
args = parser.parse_args()

rep = 'occflexi'
test = 'comp'; r = range(4, 20)
expt = args.expt; it = -1

lst_fixed_indices = [0, 1, 2, 3]

for i in r:
    for fi in lst_fixed_indices:
        subprocess.call(
            [f"scripts/tests/{test}.sh {rep} {expt} {it} {i} {fi}"],
            shell=True)
