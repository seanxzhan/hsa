import json
import numpy as np
from typing import Dict

def peek_part_meshes():
    import trimesh

    part_paths = [f"results/occflexi/occflexi_23/flexi_part_meshed_during_training/part_mesh_{i}.obj" for i in range(4)]
    meshes = [trimesh.load_mesh(part_path) for part_path in part_paths]
    concat_mesh = trimesh.util.concatenate(meshes)
    concat_mesh.export("results/occflexi/occflexi_23/flexi_part_meshed_during_training/entire_mesh.obj")

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
    # data_pt = 'data/Chair_train_new_ids_to_objs_8_0_2000.json'
    data_pt = 'data/Chair_train_new_ids_to_objs_16_0_4489.json'
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
    # np.save('data/chair_four_parts_8_0_2000.npy', indices)
    np.save('data/chair_four_parts_16_0_4489.npy', indices)


def save_four_parts_further_merged_am():
    # this gives more shapes than save_four_parts because hier is further merged
    # data_pt = 'data/Chair_train_new_ids_to_objs_8_0_2000.json'
    # data_pt = 'data/Chair_train_new_ids_to_objs_16_0_4489.json'
    # data_pt = 'data/Lamp_train_new_ids_to_objs_17_0_1554.json'
    # data_pt = 'data/Table_train_new_ids_to_objs_17_0_5707.json'
    # data_pt = 'data/Earphone_train_new_ids_to_objs_17_0_147.json'
    # data_pt = 'data/Chair_test_new_ids_to_objs_19_0_1217.json'
    data_pt = 'data/Chair_train_new_ids_to_objs_20_0_5106.json'
    with open(data_pt, 'r') as f:
        data: Dict = json.load(f)
    all_ids = np.array(list(data.keys()))

    counter = 0
    indices = []
    for id, parts in data.items():
        parts_lst = list(parts.keys())
        # if '3' in parts_lst and '4' in parts_lst and '12' in parts_lst and '15' in parts_lst:
        # chair
        if set(parts_lst) <= set(['0', '1', '4', '8']):
        # lamp
        # if set(parts_lst) <= set(['8', '1', '2', '4', '5']):
        # if set(parts_lst) <= set(['1', '2', '4', '5']):
        # table
        # if set(parts_lst) <= set(['10', '13', '2', '12', '9', '11', '4']):
        # if set(parts_lst) <= set(['13', '2', '9']):
        # # earphone
        # if set(parts_lst) <= set(['0', '5', '4']):
            indices.append(counter)
        counter += 1

    print(len(indices))
    # print(indices)
    print(all_ids[indices])
    # np.save('data/chair_four_parts_8_0_2000.npy', indices)
    # np.save('data/chair_am_four_parts_16_0_4489.npy', indices)
    # np.save('data/lamp_am_five_parts_17_0_1554.npy', indices)
    # np.save('data/table_am_three_parts_17_0_5707.npy', indices)
    # np.save('data/earphone_am_three_parts_17_0_147.npy', indices)
    np.save('data/chair_am_four_parts_19_0_1217.npy', indices)


def save_five_parts_further_merged_am():
    # this gives more shapes than save_four_parts because hier is further merged
    data_pt = 'data/Chair_train_new_ids_to_objs_20_0_5106.json'
    with open(data_pt, 'r') as f:
        data: Dict = json.load(f)
    all_ids = np.array(list(data.keys()))

    counter = 0
    indices = []
    for id, parts in data.items():
        parts_lst = list(parts.keys())
        # if '3' in parts_lst and '4' in parts_lst and '12' in parts_lst and '15' in parts_lst:
        # chair
        # if set(parts_lst) <= set(['0', '1', '4', '8']):
        # if set(parts_lst) <= set(['0', '1', '4', '8']):
        if set(parts_lst) <= set(['0', '1', '2', '4', '5', '7', '8', '9']):
        # if '9' in parts_lst:
        # lamp
        # if set(parts_lst) <= set(['8', '1', '2', '4', '5']):
        # if set(parts_lst) <= set(['1', '2', '4', '5']):
        # table
        # if set(parts_lst) <= set(['10', '13', '2', '12', '9', '11', '4']):
        # if set(parts_lst) <= set(['13', '2', '9']):
        # # earphone
        # if set(parts_lst) <= set(['0', '5', '4']):
            indices.append(counter)
        counter += 1

    print(len(indices))
    # print(indices)
    # print(all_ids[indices])
    exit(0)
    # np.save('data/chair_four_parts_8_0_2000.npy', indices)
    # np.save('data/chair_am_four_parts_16_0_4489.npy', indices)
    # np.save('data/lamp_am_five_parts_17_0_1554.npy', indices)
    # np.save('data/table_am_three_parts_17_0_5707.npy', indices)
    # np.save('data/earphone_am_three_parts_17_0_147.npy', indices)
    np.save('data/chair_am_four_parts_19_0_1217.npy', indices)


def save_four_parts_further_merged_14():
    # this gives more shapes than save_four_parts because hier is further merged
    data_pt = 'data/Chair_train_new_ids_to_objs_14_0_2000.json'
    with open(data_pt, 'r') as f:
        data: Dict = json.load(f)
    all_ids = np.array(list(data.keys()))

    counter = 0
    indices = []
    for id, parts in data.items():
        parts_lst = list(parts.keys())
        # if '3' in parts_lst and '4' in parts_lst and '12' in parts_lst and '15' in parts_lst:
        if set(parts_lst) == set(['0', '1', '2', '4']):
            indices.append(counter)
        counter += 1

    print(len(indices))
    # print(indices)
    print(all_ids[indices])
    np.save('data/chair_four_parts_14_0_2000.npy', indices)



def save_four_parts_further_merged_14_am():
    # this gives more shapes than save_four_parts because hier is further merged
    data_pt = 'data/Chair_train_new_ids_to_objs_14_0_2000.json'
    with open(data_pt, 'r') as f:
        data: Dict = json.load(f)
    all_ids = np.array(list(data.keys()))

    counter = 0
    indices = []
    for id, parts in data.items():
        parts_lst = list(parts.keys())
        # if '3' in parts_lst and '4' in parts_lst and '12' in parts_lst and '15' in parts_lst:
        if set(parts_lst) <= set(['0', '1', '2', '4']):
            indices.append(counter)
        counter += 1

    print(len(indices))
    # print(indices)
    print(all_ids[indices])
    # at most four parts
    np.save('data/chair_am_four_parts_14_0_2000.npy', indices)


def peek_11_id_to_idx(id):
    data_pt = 'data/Chair_train_new_ids_to_objs_11_0_508.json'
    with open(data_pt, 'r') as f:
        data: Dict = json.load(f)
    all_ids = list(data.keys())
    print(data_pt)
    print(f"id: {id}, idx: {all_ids.index(id)}")


def peek_occ():
    import torch
    import torch.nn.functional as F
    occ = np.load('prex_occ.npy')
    occ = torch.tensor(occ, dtype=torch.float32)
    # temperature = 1
    # soft_occ = F.sigmoid((occ - 0.5) / temperature)
    # print(soft_occ)
    # print(torch.unique(soft_occ))
    hard_occ = (occ > 0.5).float()
    print(torch.unique(hard_occ))
    np.save('prex_occ_binary.npy', hard_occ.numpy())


if __name__ == "__main__":
    # # part assembly examples
    # peek_11_id_to_idx('3344')
    # peek_11_id_to_idx('39901')
    # peek_11_id_to_idx('39440')
    # peek_11_id_to_idx('40825')
    # peek_11_id_to_idx('39901')

    # peek_11_id_to_idx('41910')
    # peek_11_id_to_idx('3344')
    # peek_11_id_to_idx('40825')
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

    # save_four_parts_further_merged()
    # save_four_parts_further_merged_am()
    save_five_parts_further_merged_am()

    # save_four_parts_further_merged_14_am()

    # peek_occ()

    # peek_part_meshes()