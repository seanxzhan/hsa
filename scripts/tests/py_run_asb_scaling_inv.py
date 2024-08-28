import subprocess
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--expt', type=int, required=True)
args = parser.parse_args()

rep = 'occflexi'
test = 'asb_scaling_inv'
expt = args.expt; it = -1

lst_anno_ids = [
    # ['38274', '39148', '42569', '2315']
    ['42213', '38274', '42586', '39148'],
    ['39194', '43121', '2728', '2315'],
]
lst = [0, 1, 2, 3]
permutations = list(itertools.permutations(lst))

for anno_ids in lst_anno_ids:
    for perm in permutations:
        subprocess.call(
            [f"scripts/tests/{test}.sh {rep} {expt} {it} "+
            f"{anno_ids[0]} {anno_ids[1]} {anno_ids[2]} {anno_ids[3]} "+
            f"{perm[0]} {perm[1]} {perm[2]} {perm[3]}"],
            shell=True)
