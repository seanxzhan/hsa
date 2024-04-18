import json
import numpy as np
from typing import Dict

data_pt = 'data/Chair_train_new_ids_to_objs_4_0_2000.json'
with open(data_pt, 'r') as f:
    data: Dict = json.load(f)
all_ids = np.array(list(data.keys()))

counter = 0
indices = []
for id, parts in data.items():
    parts_lst = list(parts.keys())
    # if '3' in parts_lst and '4' in parts_lst and '12' in parts_lst and '15' in parts_lst:
    if set(parts_lst) == set(['3', '4', '12', '15']):
        indices.append(counter)
    counter += 1

print(len(indices))
# print(indices)
print(all_ids[indices])
np.save('data/chair_four_parts_4_0_2000.npy', indices)

# data_pt = 'data/Chair_train_new_ids_to_objs_8_0_2000.json'
# with open(data_pt, 'r') as f:
#     data: Dict = json.load(f)
# all_ids = np.array(list(data.keys()))

# counter = 0
# indices = []
# for id, parts in data.items():
#     parts_lst = list(parts.keys())
#     # if '3' in parts_lst and '4' in parts_lst and '12' in parts_lst and '15' in parts_lst:
#     if set(parts_lst) == set(['0', '1', '4', '8']):
#         indices.append(counter)
#     counter += 1

# print(len(indices))
# # print(indices)
# print(all_ids[indices])
# np.save('data/chair_four_parts_8_0_2000.npy', indices)