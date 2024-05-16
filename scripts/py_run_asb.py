import os
import subprocess
import itertools

lst = [0, 1, 2, 3]
permutations = list(itertools.permutations(lst))

for perm in permutations:
    subprocess.call([f"scripts/test_local_asb.sh 66 3000 {perm[0]} {perm[1]} {perm[2]} {perm[3]}"], shell=True)