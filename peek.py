import json
import numpy as np
from typing import Dict

data_pt = 'data/Chair_train_new_ids_to_objs_4_0_2000.json'
with open(data_pt, 'r') as f:
    data: Dict = json.load(f)

counter = 0
indices = []
for id, parts in data.items():
    if '4' in list(parts.keys()):
        indices.append(counter)
    counter += 1

print(len(indices))
# print(indices)
np.save('data/chair_arm_indices_4_0_2000.npy', indices)