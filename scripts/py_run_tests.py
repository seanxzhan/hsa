import os
import subprocess
import itertools


def run_asb():
    lst = [0, 1, 2, 3]
    permutations = list(itertools.permutations(lst))

    for perm in permutations:
        subprocess.call([f"scripts/test_local_asb.sh 66 3000 {perm[0]} {perm[1]} {perm[2]} {perm[3]}"], shell=True)

def run_asb_inv():
    lst = [0, 1, 2, 3]
    permutations = list(itertools.permutations(lst))

    for perm in permutations:
        subprocess.call([f"scripts/test_local_asb_inv.sh 66 3000 {perm[0]} {perm[1]} {perm[2]} {perm[3]}"], shell=True)

def run_sample(model_idx):
    for i in range(10):
        subprocess.call([f"scripts/test_local_sample.sh 66 3000 {model_idx} {i}"], shell=True)


if __name__ == "__main__":
    run_sample(86)