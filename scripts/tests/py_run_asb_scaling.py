import subprocess
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--expt', type=int, required=True)
args = parser.parse_args()

rep = 'occflexi'
test = 'asb_scaling'
expt = args.expt; it = -1

lst_anno_ids = [
    # ['43941', '42120', '39259', '39901'],
    # ['2787', '41264', '39704', '43005'],
    # ['41378', '40825', '42312', '3144'],
    # ['38208', '3366', '49530', '37454'],
    # ['39468', '37401', '47914', '44979'],
    ['43628', '39311', '44164', '37192'],
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
