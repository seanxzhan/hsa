import os
import subprocess
import itertools

devnull = open(os.devnull, 'w')

rep = 'occflexi'
test = 'asb_scaling'
expt = 15; it = -1

# lst_anno_ids = [['43941', '42120', '39259', '39901'],
#                 ['2787', '41264', '39704', '43005']]
lst_anno_ids = [
    # ['41378', '40825', '42312', '3144'],
    ['38208', '3366', '49530', '37454'],
    ['40825', '42312', '3144', '41378']
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
