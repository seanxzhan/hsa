import json
import numpy as np
from typing import Dict

def save_four_parts():
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


def save_four_parts_further_merged():
    # this gives more shapes than save_four_parts because hier is further merged
    data_pt = 'data/Chair_train_new_ids_to_objs_8_0_2000.json'
    with open(data_pt, 'r') as f:
        data: Dict = json.load(f)
    all_ids = np.array(list(data.keys()))

    counter = 0
    indices = []
    for id, parts in data.items():
        parts_lst = list(parts.keys())
        # if '3' in parts_lst and '4' in parts_lst and '12' in parts_lst and '15' in parts_lst:
        if set(parts_lst) == set(['0', '1', '4', '8']):
            indices.append(counter)
        counter += 1

    print(len(indices))
    # print(indices)
    print(all_ids[indices])
    np.save('data/chair_four_parts_8_0_2000.npy', indices)


def peek_11_id_to_idx(id):
    data_pt = 'data/Chair_train_new_ids_to_objs_11_0_508.json'
    with open(data_pt, 'r') as f:
        data: Dict = json.load(f)
    all_ids = list(data.keys())
    print(data_pt)
    print(f"id: {id}, idx: {all_ids.index(id)}")


if __name__ == "__main__":
    # # part assembly examples
    # peek_11_id_to_idx('3344')
    # peek_11_id_to_idx('39901')
    # peek_11_id_to_idx('39440')
    # peek_11_id_to_idx('40825')
    # peek_11_id_to_idx('39901')

    # peek_11_id_to_idx('41910')
    # peek_11_id_to_idx('3344')
    peek_11_id_to_idx('40825')
    # peek_11_id_to_idx('38076')

    # peek_11_id_to_idx('43888')
    # peek_11_id_to_idx('42312')
    # peek_11_id_to_idx('3125')
    # peek_11_id_to_idx('3144')
    # peek_11_id_to_idx('44979')
    # peek_11_id_to_idx('37986')
    # peek_11_id_to_idx('2239')
    # peek_11_id_to_idx('41378')
    # peek_11_id_to_idx('42992')