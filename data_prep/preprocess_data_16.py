# part classes, includes specified number of shape with variable parts
# NOTE: obb xform is consistent throughout shapes
# NOTE: further merges some parts
# NOTE: further merges some parts
# NOTE: also get partnet surface points
# NOTE: axis aligned bounding boxes

# inherited from preprocess_data_12

import os
import json
import copy
import h5py
import torch
import kaolin
import random
import trimesh
import numpy as np
from tqdm import tqdm
from anytree import RenderTree, AnyNode, findall_by_attr, PreOrderIter
import trimesh.bounds
from utils import tree, misc, visualize, ops, transform
from utils.tree import OriNode, AMNode
from data_prep import gather_hdf5
from typing import List, Dict
import queue
from multiprocessing import Queue, Process
from torch.nn.utils.rnn import pad_sequence
from anytree.exporter import UniqueDotExporter
from collections import defaultdict
from torch_geometric.data import Data
from data_prep.partnet_pts_dataset import PartPtsDataset


name_to_cat = {
    'Chair': '03001627',
    'Lamp': '03636649'
}
cat_name = 'Chair'
cat_id = name_to_cat[cat_name]

pt_sample_res = 64

partnet_dir = '/datasets/PartNet'
partnet_index_path = '/sota/partnet_dataset/stats/all_valid_anno_info.txt'
stats_dir = '/sota/partnet_dataset/stats'
index_path = stats_dir + '/all_valid_anno_info.txt'
train_models_path = stats_dir + f'/train_val_test_split/{cat_name}.train.json'
val_models_path = stats_dir + f'/train_val_test_split/{cat_name}.val.json'
test_models_path = stats_dir + f'/train_val_test_split/{cat_name}.test.json'
after_merge_label_ids = stats_dir + f'/after_merging_label_ids/{cat_name}.txt'
OBB_REP_SIZE = 15

train_f = open(train_models_path); val_f = open(val_models_path); test_f = open(test_models_path)
train_ids = json.load(train_f); val_ids = json.load(val_f); test_ids = json.load(test_f)

# print(len(train_ids))
# exit(0)

# -------- get tableau colors for visualization -------- 
all_colors = np.array(visualize.get_tab_20_saturated(50))[:20]
np.random.seed(12345)
color_indices = np.arange(len(all_colors))
np.random.shuffle(color_indices)
all_colors = all_colors[color_indices]
all_colors = all_colors.tolist() * 3
all_colors = np.array(all_colors)


def align_xform(ext, xform):
    # takes old ext and xform and align the coordinate frames such that the 
    # y axis points in world up (0, 1, 0), x axis points in world right (1, 0, 0)
    # world axes
    ref_x_axis = np.array([1, 0, 0])
    ref_z_axis = np.array([0, 1, 0])

    orig_x_axis = xform[:3, :3][:, 0]
    orig_y_axis = xform[:3, :3][:, 1]
    orig_z_axis = xform[:3, :3][:, 2]

    lst_axes = [orig_x_axis, orig_y_axis, orig_z_axis]
    lst_exts = [ext[0], ext[1], ext[2]]
    
    # book keep new extents and xforms
    new_ext = np.array([0, 0, 0], dtype=np.float32)
    new_xform = np.eye(4)
    new_xform[:3, 3] = xform[:3, 3]     # set translation
    
    # first, get the axis that aligns with world z (up)
    dot_out = []
    for ax in lst_axes:
        dot_out.append(np.abs(np.dot(ax, ref_z_axis)))
    which_aligns = np.argmax(dot_out)
    if np.dot(lst_axes[which_aligns], ref_z_axis) >= 0:
        new_xform[:3, 1] = lst_axes[which_aligns]
    else:
        new_xform[:3, 1] = -lst_axes[which_aligns]
    new_ext[1] = lst_exts[which_aligns]
    del lst_axes[which_aligns]
    del lst_exts[which_aligns]

    # next, get the axis that aligns with world x (right)
    dot_out = []
    for ax in lst_axes:
        dot_out.append(np.abs(np.dot(ax, ref_x_axis)))
    which_aligns = np.argmax(dot_out)
    if np.dot(lst_axes[which_aligns], ref_x_axis) >= 0:
        new_xform[:3, 0] = lst_axes[which_aligns]
    else:
        new_xform[:3, 0] = -lst_axes[which_aligns]
    new_ext[0] = lst_exts[which_aligns]
    del lst_axes[which_aligns]
    del lst_exts[which_aligns]

    # take from what's left in the axes and exts lists
    new_xform[:3, 2] = lst_axes[0]
    new_ext[2] = lst_exts[0]

    # check handedness, flip an axis if necessary
    if np.dot(np.cross(new_xform[:3, 0], new_xform[:3, 1]),
                new_xform[:3, 2]) < 0:
        new_xform[:3, 2] = -new_xform[:3, 2]

    # ext_xform = (ext, xform,
    #              orig_x_axis,
    #              orig_y_axis,
    #              orig_z_axis,
    #              xform[:3, 3])
    ext_xform = (new_ext, new_xform,
                    new_xform[:3, 0],
                    new_xform[:3, 1],
                    new_xform[:3, 2],
                    new_xform[:3, 3])
    
    return ext_xform


def build_obbs(anno_id, part_info: Dict):
    # first, normalize the mesh vertices
    all_mesh_verts = []
    all_mesh_faces = []
    all_mesh_parts = []
    part_vert_nums = []
    subpart_num_meshes = []
    partnet_objs_dir = os.path.join(partnet_dir, anno_id, 'objs')
    for name, subparts in part_info.items():
        # part_c = 0
        for subpart in subparts:
            obj_names = subpart['objs']
            obj_names = [x+'.obj' for x in obj_names]
            obj_paths = [os.path.join(partnet_objs_dir, x) for x in obj_names]
            meshes: List[trimesh.Trimesh] =\
                [trimesh.load_mesh(x) for x in obj_paths]
            all_mesh_parts += meshes
            subpart_num_meshes.append(len(meshes))
            for m in meshes:
                mesh_verts = m.vertices
                mesh_faces = m.faces
                all_mesh_verts.append(mesh_verts)
                all_mesh_faces.append(mesh_faces)
                part_vert_nums.append(len(mesh_verts))
    
    entire_mesh_verts = np.concatenate(all_mesh_verts)
    entire_mesh_verts = torch.from_numpy(entire_mesh_verts).to(torch.float32)
    entire_mesh_verts = entire_mesh_verts.unsqueeze(0)
    entire_mesh_verts = kaolin.ops.pointcloud.center_points(entire_mesh_verts,
                                                            normalize=True)
    entire_mesh_verts = entire_mesh_verts.numpy()[0]

    # make part meshes from the normalized entire mesh
    part_meshes = []
    up_to_now = 0
    for i in range(len(part_vert_nums)):
        curr_num_verts = part_vert_nums[i]
        verts = entire_mesh_verts[up_to_now:up_to_now+curr_num_verts]
        up_to_now += curr_num_verts
        part_meshes.append(trimesh.Trimesh(vertices=verts,
                                           faces=all_mesh_faces[i]))

    # part_info: new_am_nodes_modified dictionary
    obbs = []
    name_to_obbs = {}
    count = 0
    # all_mesh_parts = []
    up_to_now = 0
    for name, subparts in part_info.items():
        meshes = []
        for subpart in subparts:
            curr_num_meshes = subpart_num_meshes[count]
            meshes += part_meshes[up_to_now:up_to_now+curr_num_meshes]
            up_to_now += curr_num_meshes
            count += 1
        mesh = trimesh.util.concatenate(meshes)
        obb: trimesh.primitives.Box = mesh.bounding_box
        ext = np.array(obb.primitive.extents)
        xform = np.array(obb.primitive.transform)
        ext_xform = align_xform(ext, xform)
        obbs.append(ext_xform)
        name_to_obbs[name] = ext_xform
    
    entire_mesh_orig = trimesh.util.concatenate(all_mesh_parts)

    return obbs, entire_mesh_orig, name_to_obbs


def condense_obbs(obbs):
    # given a list of oriented bounding boxes, make them into compact rep.
    n_boxes = len(obbs)
    compact = torch.zeros((n_boxes, OBB_REP_SIZE), dtype=torch.float32)
    for i in range(n_boxes):
        obb = obbs[i]
        compact[i][:3] = torch.from_numpy(np.array(obb[0])).to(torch.float32)
        xform = obb[1]
        rot = xform[:3, :3].flatten()
        xlation = xform[:3, 3].flatten()
        compact[i][3:12] = torch.from_numpy(np.array(rot)).to(torch.float32)
        compact[i][12:] = torch.from_numpy(np.array(xlation)).to(torch.float32)
    return compact


def build_dense_graph(union_root: AnyNode, name_to_obbs: Dict, obbs,
                      num_union_nodes_class, all_unique_names,
                      shape_root):
    num_unique_part_classes = len(all_unique_names)

    # node features, len = num_union_nodes
    node_features = []
    # xforms
    xforms = [np.eye(4, dtype=np.float32)[None, :]] * num_unique_part_classes
    # extents
    extents = [np.zeros((1, 3), dtype=np.float32)] * num_unique_part_classes
    # for each unique part, whare are the valid nodes in this specific shape
    part_nodes = np.zeros((num_unique_part_classes, num_union_nodes_class))
    # for building edges
    nodes = []
    adj = np.zeros((num_union_nodes_class, num_union_nodes_class))

    shape_part_names = list(name_to_obbs.keys())

    def tree_to_lst(node):
        name = node.name
        nodes.append(node)
        if name not in shape_part_names:
            # union node class that can't be found in shape
            node_features.append(np.zeros((1, OBB_REP_SIZE), dtype=np.float32))
        else:
            obb = name_to_obbs[name]
            condensed_obb = condense_obbs([obb]).numpy()
            node_features.append(condensed_obb)
            row_i_part_nodes = all_unique_names.index(name)
            part_nodes[row_i_part_nodes, len(nodes) - 1] = 1

            extents[row_i_part_nodes] = np.array(obb[0]).reshape((1, 3))
            xforms[row_i_part_nodes] = np.linalg.inv(np.array(obb[1]).reshape((1, 4, 4)))
        for child in node.children:
            tree_to_lst(child)

    tree_to_lst(union_root)
    node_features = np.concatenate(node_features, axis=0)
    xforms = np.concatenate(xforms, axis=0)
    extents = np.concatenate(extents, axis=0)

    # this creates the accurate adjacency matrix for each shape
    def make_edges_shape(node):
        node_name = node.name
        for child in node.children:
            child_name = child.name
            node_name_in = findall_by_attr(shape_root, node_name, name='name')
            child_name_in = findall_by_attr(shape_root, child_name, name='name')
            if node_name_in and child_name_in:
                adj[nodes.index(node), nodes.index(child)] = 1
            make_edges_shape(child)

    make_edges_shape(union_root)

    # this creates the adjacency matrix of the union graph
    # (this will give the same adjacency matrix regardless of shape)
    def make_edges(node):
        for child in node.children:
            adj[nodes.index(node), nodes.index(child)] = 1
            make_edges(child)

    node_features = node_features[None, :]
    adj = adj[None, :]
    part_nodes = part_nodes[None, :]

    # relation information

    # connectivity = np.zeros((num_unique_part_classes, num_unique_part_classes))
    # connectivity[0, 1] = 1
    # connectivity[0, 2] = 1
    # connectivity[1, 2] = 1
    # connectivity[2, 3] = 1
    # connected_xforms = connectivity @ xforms[:, :3, 3]

    connectivity = [(0, 1), (0, 2), (1, 2), (2, 3)]
    relations = np.zeros((len(connectivity), 4, 4), dtype=np.float32)
    
    for i, conn in enumerate(connectivity):
        src = xforms[conn[0]]
        tgt = xforms[conn[1]]
        tgt_in_src = np.linalg.inv(src) @ tgt
        relations[i] = tgt_in_src

    return node_features, adj, part_nodes, xforms, extents, relations


def merge_partnet_after_merging(anno_id, info=False):
    """Resolves inconsistencies within partnet
    1: result_after_merging misses labels for certain parts
        (parts exist, but annotation doesn't)
        (example: 2297 yoke is missing annotation)
    2: partnet doesn't have information for which parts got merged
        (example: 38725 seat_frame_slant_bar --> seat_frame)
    """    
    # ori: original, before merging
    ori_nodes, ori_id_to_ori_nodes_idx =\
        tree.build_tree_from_json(
            os.path.join(partnet_dir, anno_id, 'result.json'))
    # am: after merging
    am_nodes, am_id_to_am_nodes_idx =\
        tree.build_tree_from_json_after_merge(
            os.path.join(partnet_dir, anno_id, 'result_after_merging.json'))

    # solution to issue 1:
    # for each node n, if objs list has more than all objs lists of children,
    # merge children and n into a leaf node, using the name and id of n
    # also outputs a map that keeps track of the change {new_id: old_ids}
    def check_objs(node: AnyNode):
        """Check if a node's objs list is the sum of all children's lists
        """
        all_objs = node.objs
        all_children_objs = set()
        all_children_ids = []

        def traverse(n: AnyNode, all_children_objs, all_children_ids):
            all_children_objs.update(n.objs)
            all_children_ids.append(n.id)
            if n.children:
                for child in n.children:
                    traverse(child, all_children_objs, all_children_ids)
        
        if node.children:
            for child in node.children:
                traverse(child, all_children_objs, all_children_ids)
            matched = set(all_objs) == all_children_objs
        else:
            matched = True

        return matched,\
            all_children_objs, all_children_ids

    # new tree nodes
    new_am_nodes = []
    # key: am node id, value: am node ids that are merged with key
    # grouping_map will be used to group partnet pcd based on point labels
    grouping_map = {}
    unique_part_names = []
    num_unmatched = 0

    def build_new_tree(node: AMNode, parent: AMNode):
        """TODO: for 37825, the star_leg_base includes everything because
        seat_connector is missing and yokes are missing. Also, each wheel
        isn't attached to each leg...
        examples: 37825, 2297
        """
        nonlocal num_unmatched
        objs_matched_up, _, children_ids = check_objs(node)
        new_node = AMNode(ori_id=node.ori_id,
                          id=node.id,
                          objs=node.objs,
                          name=node.name)
        new_node.parent = parent
        new_am_nodes.append(new_node)
        if objs_matched_up:
            if not node.is_leaf_node():
                for child in node.children:
                    build_new_tree(child, new_node)
            else:
                unique_part_names.append(node.name)\
                    if node.name not in unique_part_names else None
        else:
            grouping_map[node.id] = children_ids
            unique_part_names.append(node.name)\
                if node.name not in unique_part_names else None

    build_new_tree(am_nodes[0], None)
    unique_part_names = sorted(unique_part_names)

    # parts_info maps unique part names to id, ori_id(s), and objs
    parts_info = {}
    for pn in unique_part_names:
        parts_info[pn] = []

    def build_parts_info_from_tree(node: AMNode):
        if node.is_leaf_node():
            packet = {}
            packet['id'] = node.id
            packet['ori_ids'] = [node.ori_id]
            packet['objs'] = node.objs
            parts_info[node.name].append(packet)
        else:
            for child in node.children:
                build_parts_info_from_tree(child)
    
    build_parts_info_from_tree(new_am_nodes[0])

    # with open(f'data_prep/tmp/{anno_id}_parts_info_before.json', 'w') as f:
    #     json.dump(parts_info, f)

    # as of now, issue 1 is resolved. now on to issue 2

    # solution to issue 2:
    # for each leaf node in new_am_nodes, check if ori_node n with the same
    # ori_id has more children. if so, modify parts_info by appending 
    # n's children to ori_id(s) list
    def update_parts_info(node: AMNode):
        """takes in node from new_am_tree
        """
        if node.is_leaf_node():
            idx_of_ori_node = ori_id_to_ori_nodes_idx[node.ori_id]
            ori_node: OriNode = ori_nodes[idx_of_ori_node]
            if not ori_node.is_leaf_node():
                children_ori_ids = ori_node.get_ids_of_all_children()
                lst_parts_w_name = parts_info[node.name]
                for i, p in enumerate(lst_parts_w_name):
                    if p['id'] == node.id:
                        packet = copy.deepcopy(p)
                        packet["ori_ids"] = children_ori_ids
                        parts_info[node.name][i] = packet
        else:
            for child in node.children:
                update_parts_info(child)
    
    update_parts_info(new_am_nodes[0])

    root_node: AMNode = new_am_nodes[0]

    if info:
        print(RenderTree(root_node))
        with open(f'data_prep/tmp/{anno_id}_parts_info.json', 'w') as f:
            json.dump(parts_info, f)

    with open('data_prep/further_merge_info_8.json', 'r') as f:
    # with open(f'data_prep/{cat_name}_further_merge_info_16.json', 'r') as f:
        further_merge_info = json.load(f)
    further_merge_parents = list(further_merge_info.keys())
    further_merge_children = []
    for p, kids in further_merge_info.items():
        further_merge_children += kids

    merged_parts_info = {}
    parts_info_keys = list(parts_info.keys())

    def further_merge_parts(merged_root_node: AMNode,
                            node: AMNode):
        """merges smaller parts together
        effectively reconstruct a tree while skipping some nodes
        """
        parent = findall_by_attr(merged_root_node, node.parent.id, name="id")[0]
        node_to_add = AMNode(ori_id=node.ori_id,
                             id=node.id,
                             objs=node.objs,
                             name=node.name)
        node_to_add.parent = parent

        if node.name not in further_merge_parents:
            # node isn't a part that needs to have children merged
            if node.name in parts_info_keys:
                packet: List = parts_info[node.name]
                this_node_info = list(filter(lambda x: x["id"] == node.id,
                                             packet))
                if node.name not in list(merged_parts_info.keys()):
                    merged_parts_info[node.name] = this_node_info
                else:
                    merged_parts_info[node.name] += this_node_info
            for child in node.children:
                further_merge_parts(merged_root_node, child)
        else:
            # if node.is_leaf_node():
            #     print(node.name)
            #     merged_parts_info[node.name] = parts_info[node.name]
            #     return

            # node is a part that needs to have children merged
            node_leaf_descs = list(filter(lambda x: x.is_leaf_node(),
                                          node.descendants))
            # assumes that all leaf nodes under a mergable parent are merged
            accum_part_info = []
            if node.name in list(parts_info.keys()):
                packet = parts_info[node.name]
                this_node_info = list(filter(lambda x: x["id"] == node.id, packet))
                accum_part_info += this_node_info
            for desc in node_leaf_descs:
                packet: List = parts_info[desc.name]
                this_desc_info = list(filter(lambda x: x["id"] == desc.id,
                                             packet))
                accum_part_info += this_desc_info
            if node.name not in list(merged_parts_info.keys()):
                merged_parts_info[node.name] = accum_part_info
            else:
                merged_parts_info[node.name] += accum_part_info
    
    merged_root_node = AMNode(ori_id=root_node.ori_id,
                              id=root_node.id,
                              objs=root_node.objs,
                              name=root_node.name,)
    for child in root_node.children:
        further_merge_parts(merged_root_node, child)
    # merged_root_node = root_node

    if info:
        with open(f'data_prep/tmp/{anno_id}_merged_parts_info.json', 'w') as f:
            json.dump(merged_parts_info, f)
        print(RenderTree(merged_root_node))
    
    if merged_parts_info == {}:
        valid = False
        return None, None, None, None, None, None, anno_id, valid

    # -------- build bounding boxes --------
    obbs, entire_mesh, name_to_obbs = build_obbs(anno_id, merged_parts_info)

    def complete_hier_to_part_class_hier(new_root_node, node_to_add):
        """only add to new node if the name is new in new_root_node
        this effectively returns a new tree where each node in the tree
        represents a part class rather than a part instance
        """
        tmp = findall_by_attr(new_root_node, node_to_add.name, name="name")
        if not tmp:
            new_parent = findall_by_attr(new_root_node,
                                         node_to_add.parent.name,
                                         name="name")[0]
            if node_to_add.name in list(name_to_obbs.keys()):
                obb = name_to_obbs[node_to_add.name]
            else:
                obb = None
            new_node = AnyNode(name=node_to_add.name, obb=obb)
            new_node.parent = new_parent
        for child in node_to_add.children:
            complete_hier_to_part_class_hier(new_root_node, child)

    new_root_node = AnyNode(name="chair", obb=None)
    # new_root_node = AnyNode(name="lamp", obb=None)
    complete_hier_to_part_class_hier(new_root_node, merged_root_node)

    if info:
        # print(RenderTree(new_root_node))
        for pre, _, node in RenderTree(new_root_node):
            print("%s%s" % (pre, node.name))

    # map from name to list of ori_ids and list of objs
    name_to_ori_ids_and_objs = {}
    for name, lst_of_parts in merged_parts_info.items():
        packet = {}
        packet['ori_ids'] = []
        packet['objs'] = []
        for p in lst_of_parts:
            packet['ori_ids'] += p['ori_ids']
            packet['objs'] += p['objs']
        name_to_ori_ids_and_objs[name] = packet

    if info:
        with open(f'data_prep/tmp/{anno_id}_part_map.json', 'w') as f:
            json.dump(name_to_ori_ids_and_objs, f)

    unique_part_names = sorted(list(merged_parts_info.keys()))

    valid = True
    return unique_part_names, name_to_ori_ids_and_objs,\
        obbs, entire_mesh, name_to_obbs, new_root_node,\
        anno_id, valid


def merge_partnet_after_merging_mp(q: Queue, id_pairs):
    for id_pair in id_pairs:
        q.put(merge_partnet_after_merging(id_pair['anno_id']))


def export_data(split_ids: Dict, save_data=True, start=0, end=0,
                get_part_distr=False):
    """Exports an hdf5 file containing:
    - part_num_indices
    - all_indices
    - gt_flags
    - normalized_points
    - values
    where part_num_indices and all_indices can be used to reconstruct 
    fg_part_indices with convert_fg_part_indices_to_flat_list and 
    convert_flat_list_to_gf_part_indices
    """
    dict_each_shape_unique_names = {}
    all_unique_names = set()
    dict_all_names_to_ori_ids_and_objs = {}
    dict_all_obbs = {}
    dict_all_entire_meshes = {}
    dict_all_name_to_obbs = {}
    dict_all_root_nodes = {}
    dict_all_valid_anno_ids = {}

    def set_dict(book, id_pairs):
        for pair in id_pairs:
            anno_id = pair['anno_id']
            book[anno_id] = None

    set_dict(dict_each_shape_unique_names, split_ids[start:end])
    set_dict(dict_all_names_to_ori_ids_and_objs, split_ids[start:end])
    set_dict(dict_all_obbs, split_ids[start:end])
    set_dict(dict_all_entire_meshes, split_ids[start:end])
    set_dict(dict_all_name_to_obbs, split_ids[start:end])
    set_dict(dict_all_root_nodes, split_ids[start:end])
    set_dict(dict_all_valid_anno_ids, split_ids[start:end])

    num_workers = 1
    list_of_lists = misc.chunks(split_ids[start:end], num_workers)
    q = Queue()
    workers = [
        Process(target=merge_partnet_after_merging_mp, args=(q, lst))
        for lst in list_of_lists]
    for p in workers:
        p.start()
    pbar = tqdm(total=len(split_ids[start:end]))
    while True:
        flag = True
        try:
            unique_names, part_map, obbs, entire_mesh, name_to_obbs, root_node,\
                anno_id, valid = q.get(True, 1.0)
        except queue.Empty:
            flag = False
        if flag:
            if valid:
                dict_each_shape_unique_names[anno_id] = unique_names
                all_unique_names.update(set(unique_names))
                dict_all_names_to_ori_ids_and_objs[anno_id] = part_map
                dict_all_obbs[anno_id] = obbs
                dict_all_entire_meshes[anno_id] = entire_mesh
                dict_all_name_to_obbs[anno_id] = name_to_obbs
                dict_all_root_nodes[anno_id] = root_node
                dict_all_valid_anno_ids[anno_id] = True
            else:
                dict_all_valid_anno_ids[anno_id] = False
                print("invalid shape: ", anno_id)
            pbar.update(1)
        all_exited = True
        for p in workers:
            if p.exitcode is None:
                all_exited = False
                break
        if all_exited and q.empty():
            break
    pbar.close()
    for p in workers:
        p.join()
    all_unique_names = sorted(list(all_unique_names))
    num_parts = len(all_unique_names)

    each_shape_unique_names = []
    all_names_to_ori_ids_and_objs = []
    all_obbs = []
    all_entire_meshes = []
    all_name_to_obbs = []
    all_root_nodes = []
    all_valid_anno_ids = []

    for k, v in dict_all_valid_anno_ids.items():
        if v:
            each_shape_unique_names.append(dict_each_shape_unique_names[k])
            all_names_to_ori_ids_and_objs.append(dict_all_names_to_ori_ids_and_objs[k])
            all_obbs.append(dict_all_obbs[k])
            all_entire_meshes.append(dict_all_entire_meshes[k])
            all_name_to_obbs.append(dict_all_name_to_obbs[k])
            all_root_nodes.append(dict_all_root_nodes[k])
            all_valid_anno_ids.append(k)

    print("number of valid shapes: ", len(all_valid_anno_ids))
    
    num_shapes = len(all_valid_anno_ids)

    # make frequency map
    name_freq = {}
    for un in all_unique_names:
        name_freq[un] = 0
    for esun in each_shape_unique_names:
        for un in esun:
            name_freq[un] += 1
    freq = []
    for _, v in name_freq.items():
        freq.append(v)
    part_distr = torch.nn.functional.softmax(torch.Tensor(freq), dim=0)
    part_distr = part_distr.numpy()
    if get_part_distr:
        return part_distr

    unique_name_to_new_id = {}
    for i, un in enumerate(all_unique_names):
        unique_name_to_new_id[un] = i
    with open(
        f'data/{cat_name}_part_name_to_new_id_16_{start}_{end}.json',
        'w') as f:
        json.dump(unique_name_to_new_id, f)
    
    all_ori_ids_to_new_ids = []
    all_new_ids_to_objs = {}
    # TODO: get part id to number of models with part (frequency map)
    for i, part_map in enumerate(all_names_to_ori_ids_and_objs):
        ori_ids_to_new_ids = {}
        new_ids_to_objs = {}
        for part_name, packet in part_map.items():
            ori_ids = packet['ori_ids']
            objs = packet['objs']
            new_id_for_part = unique_name_to_new_id[part_name]
            for oi in ori_ids:
                ori_ids_to_new_ids[oi] = new_id_for_part
            new_ids_to_objs[new_id_for_part] = objs
        all_ori_ids_to_new_ids.append(ori_ids_to_new_ids)
        all_new_ids_to_objs[all_valid_anno_ids[i]] = new_ids_to_objs

    with open(
        f'data/{cat_name}_train_new_ids_to_objs_16_{start}_{end}.json',
        'w') as f:
        json.dump(all_new_ids_to_objs, f)

    # # union of trees, based on part classes
    union_root_part = AnyNode(name='chair')
    # union_root_part = AnyNode(name='lamp')
    for root in all_root_nodes:
        tree.traverse_and_add_class(union_root_part, root)
    UniqueDotExporter(union_root_part,
                      indent=0,
                      nodenamefunc=lambda node: node.name,
                      nodeattrfunc=lambda node: "shape=box",).to_picture(
                          f"data_prep/tmp/tree_union_class_16_{start}_{end}.png")
    num_union_nodes_class = sum(1 for _ in PreOrderIter(union_root_part))
    print("num_union_nodes_class: ", num_union_nodes_class)

    nodes = []
    union_node_names = []
    def tree_to_list(node):
        nodes.append(node)
        union_node_names.append(node.name)
        for child in node.children:
            tree_to_list(child)
    tree_to_list(union_root_part)
    adj = np.zeros((num_union_nodes_class, num_union_nodes_class))
    def make_edges(node):
        for child in node.children:
            adj[nodes.index(node), nodes.index(child)] = 1
            make_edges(child)
    make_edges(union_root_part)

    np.save(f'data/{cat_name}_union_node_names_16_{start}_{end}.npy',
            union_node_names)
    
    # reconstruct a tree from adj
    recon_root = tree.recon_tree(adj, union_node_names)
    UniqueDotExporter(recon_root,
                      indent=0,
                      nodenamefunc=lambda node: node.name,
                      nodeattrfunc=lambda node: "shape=box",
                      ).to_picture(
                          f"data_prep/tmp/recon_tree_union_16_{start}_{end}.png")

    print("making dense graphs")
    all_node_features = []
    all_adj = []
    all_part_nodes = []
    all_xforms = []
    all_extents = []
    all_relations = []
    for i in tqdm(range(num_shapes)):
        # print(f"------------ {i} ------------")
        node_features, adj, part_nodes,\
            xforms, extents, relations =\
            build_dense_graph(union_root_part,
                              all_name_to_obbs[i],
                              all_obbs[i],
                              num_union_nodes_class,
                              all_unique_names,
                              all_root_nodes[i])
        all_node_features.append(node_features)
        all_adj.append(adj)
        all_part_nodes.append(part_nodes)
        all_xforms.append(xforms)
        all_extents.append(extents)
        all_relations.append(relations)
    

    if not save_data:
        return unique_name_to_new_id, all_entire_meshes, all_ori_ids_to_new_ids,\
            all_obbs, all_name_to_obbs
    
    # exit(0)

    fn = f'data/{cat_name}_train_{pt_sample_res}_16_{start}_{end}.hdf5'
    hdf5_file = h5py.File(fn, 'w')
    hdf5_file.create_dataset(
        'part_num_indices', [num_shapes, num_parts],
        dtype=np.int64)
    hdf5_file.create_dataset(
        'all_indices', [num_shapes, ],
        dtype=h5py.vlen_dtype(np.int64))
    hdf5_file.create_dataset(
        'normalized_points', [num_shapes, (pt_sample_res/2)**3, 3],
        dtype=np.float32)
    hdf5_file.create_dataset(
        'values', [num_shapes, (pt_sample_res/2)**3, 1],
        dtype=np.float32)
    hdf5_file.create_dataset(
        'node_features', [num_shapes, num_union_nodes_class, OBB_REP_SIZE],
        dtype=np.float32)
    hdf5_file.create_dataset(
        'adj', [num_shapes, num_union_nodes_class, num_union_nodes_class],
        dtype=np.float32)
    hdf5_file.create_dataset(
        'part_nodes', [num_shapes, num_parts, num_union_nodes_class],
        dtype=np.int64)
    hdf5_file.create_dataset(
        'xforms', [num_shapes, num_parts, 4, 4],
        dtype=np.float32)
    hdf5_file.create_dataset(
        'extents', [num_shapes, num_parts, 3],
        dtype=np.float32)
    hdf5_file.create_dataset(
        'transformed_points', [num_shapes, num_parts, (pt_sample_res/2)**3, 3],
        dtype=np.float32)
    hdf5_file.create_dataset(
        'empty_parts', [num_shapes, (pt_sample_res/2)**3, num_parts],
        dtype=np.uint8)
    hdf5_file.create_dataset(
        'relations', [num_shapes, 4, 4, 4],
        dtype=np.float32)

    # all_pts_data = [None] * num_shapes
    # all_pts_whole_data = [None] * num_shapes

    # valid_anno_id_to_idx = {}
    # for i, x in enumerate(all_valid_anno_ids):
    #     valid_anno_id_to_idx[x] = i

    # # lol: list of lists 
    # num_workers = 2
    # lol_anno_ids = misc.chunks(all_valid_anno_ids, num_workers)
    # lol_entire_meshes = misc.chunks(all_entire_meshes, num_workers)
    # lol_ori_ids_to_new_ids = misc.chunks(all_ori_ids_to_new_ids, num_workers)

    # n_lists = len(lol_anno_ids)
    # assert n_lists == len(lol_entire_meshes)
    # assert n_lists == len(lol_ori_ids_to_new_ids)

    # q = Queue()
    # workers = [
    #     Process(target=make_data_for_one_mp,
    #             args=(q, lol_anno_ids[i],
    #                   unique_name_to_new_id,
    #                   lol_entire_meshes[i],
    #                   lol_ori_ids_to_new_ids[i]))
    #     for i in range(n_lists)
    # ]
    # for p in workers:
    #     p.start()

    # # print(f"creating dataset: {fn}")
    # pbar = tqdm(total=len(all_valid_anno_ids))
    # while True:
    #     flag = True
    #     try:
    #         anno_id, out = q.get(True, 1.0)
    #     except queue.Empty:
    #         flag = False
    #     if flag:
    #         idx = valid_anno_id_to_idx[anno_id]
    #         # hdf5_file['part_num_indices'][idx] = out['part_num_indices']
    #         # hdf5_file['all_indices'][idx] = out['all_indices']
    #         # hdf5_file['normalized_points'][idx] = out['normalized_points']
    #         # hdf5_file['values'][idx] = out['values']
    #         # hdf5_file['node_features'][idx] = all_node_features[idx]
    #         # hdf5_file['adj'][idx] = all_adj[idx]
    #         # hdf5_file['part_nodes'][idx] = all_part_nodes[idx]
    #         # hdf5_file['xforms'][idx] = all_xforms[idx]
    #         # hdf5_file['extents'][idx] = all_extents[idx]
    #         # all_pts_data[idx] = out['pts_data']
    #         all_pts_whole_data[idx] = out['pts_whole_data']

    #         # for p in range(num_parts):
    #         #     xform = all_xforms[idx][p]
    #         #     pts = out['normalized_points'][0]
    #         #     hdf5_file['transformed_points'][idx, p] =\
    #         #         transform.transform_points(pts, xform)
    #         # hdf5_file['empty_parts'][idx] = 1-torch.from_numpy(
    #         #     all_part_nodes[idx][0]).sum(1).unsqueeze(1).expand(
    #         #         -1, len(pts)).transpose(0, 1).numpy()
            
    #         pbar.update(1)
    #     all_exited = True
    #     for p in workers:
    #         if p.exitcode is None:
    #             all_exited = False
    #             break
    #     if all_exited and q.empty():
    #         break
    # pbar.close()
    # for p in workers:
    #     p.join()

    # # hdf5_file.close()
    
    # # pyg = PartPtsDataset(f'data/{cat_name}_train_{pt_sample_res}_16_{start}_{end}',
    # #                      all_pts_data)
    # # pyg.process()

    # pyg = PartPtsDataset(f'data/{cat_name}_train_{pt_sample_res}_16_{start}_{end}_whole',
    #                      all_pts_whole_data)
    # pyg.process()

    all_pts_data = []
    all_pts_whole_data = []

    # print(f"creating dataset: {fn}")
    for i, anno_id in enumerate(tqdm(all_valid_anno_ids)):
        # print(anno_id)
        out = make_data_for_one(anno_id, unique_name_to_new_id,
                                all_entire_meshes[i],
                                all_ori_ids_to_new_ids[i])
        hdf5_file['part_num_indices'][i] = out['part_num_indices']
        hdf5_file['all_indices'][i] = out['all_indices']
        hdf5_file['normalized_points'][i] = out['normalized_points']
        hdf5_file['values'][i] = out['values']
        hdf5_file['node_features'][i] = all_node_features[i]
        hdf5_file['adj'][i] = all_adj[i]
        hdf5_file['part_nodes'][i] = all_part_nodes[i]
        hdf5_file['xforms'][i] = all_xforms[i]
        hdf5_file['extents'][i] = all_extents[i]
        hdf5_file['relations'][i] = all_relations[i]
        all_pts_data.append(out['pts_data'])
        all_pts_whole_data.append(out['pts_whole_data'])

        for p in range(num_parts):
            xform = all_xforms[i][p]
            pts = out['normalized_points'][0]
            hdf5_file['transformed_points'][i, p] =\
                transform.transform_points(pts, xform)
        hdf5_file['empty_parts'][i] = 1-torch.from_numpy(
            all_part_nodes[i][0]).sum(1).unsqueeze(1).expand(
                -1, len(pts)).transpose(0, 1).numpy()

    hdf5_file.close()

    pyg = PartPtsDataset(f'data/{cat_name}_train_{pt_sample_res}_16_{start}_{end}',
                         all_pts_data)
    pyg.process()

    pyg = PartPtsDataset(f'data/{cat_name}_train_{pt_sample_res}_16_{start}_{end}_whole',
                         all_pts_whole_data)
    pyg.process()



def make_data_for_one_mp(q: Queue, anno_ids, unique_name_to_new_id,
                         entire_meshes, ori_ids_to_new_ids_list):
    for i, anno_id in enumerate(anno_ids):
        q.put((anno_id, make_data_for_one(anno_id,
                                unique_name_to_new_id,
                                entire_meshes[i],
                                ori_ids_to_new_ids_list[i])))


def make_data_for_one(anno_id,
                      unique_name_to_new_id: Dict,
                      entire_mesh,
                      ori_ids_to_new_ids,
                      vis=False):
    unique_names = list(unique_name_to_new_id.keys())

    partnet_point_sample_dir = os.path.join(partnet_dir, anno_id, 'point_sample')
    partnet_points_ply = os.path.join(partnet_point_sample_dir, 'ply-10000.ply')
    labels_path = os.path.join(partnet_point_sample_dir, 'label-10000.txt')
    orig_labels, _, _, _ = ops.parse_labels_txt(labels_path)

    # print(ori_ids_to_new_ids)

    new_labels = []
    for ol in orig_labels:
        new_labels.append(ori_ids_to_new_ids[ol])
    new_labels = np.array(new_labels)

    # return None

    partnet_pcd_orig: trimesh.points.PointCloud = trimesh.load(partnet_points_ply)

    # print(partnet_pcd_orig.shape)

    fg_voxels, fg_binvox_xform = get_info_from_voxels(
        anno_id, pt_sample_res, entire_mesh)
    partnet_pcd_in_fg_vox_grid = transform.mesh_space_to_voxel_space_centered(
        fg_binvox_xform, partnet_pcd_orig.vertices)
    
    partnet_pcd_part_points = []
    partnet_pcd_part_labels = []
    for un in unique_names:
        new_id = unique_name_to_new_id[un]
        if new_id not in new_labels:
            continue
        part_indices = np.argwhere(new_labels == new_id)[:,0]
        part_points = partnet_pcd_in_fg_vox_grid[part_indices]
        part_points = kaolin.ops.pointcloud.center_points(
            torch.from_numpy(part_points).unsqueeze(0), normalize=True)
        part_points = part_points[0].to(torch.float32)
        partnet_pcd_part_points.append(part_points)
        partnet_pcd_part_labels.append(np.repeat([new_id], len(part_points)))

    partnet_pcd_part_points = np.concatenate(partnet_pcd_part_points, axis=0)
    partnet_pcd_part_labels = np.concatenate(partnet_pcd_part_labels, axis=0)
    
    pts_data = Data(pos=partnet_pcd_part_points,
                    part_label=partnet_pcd_part_labels)

    partnet_pcd_norm = kaolin.ops.pointcloud.center_points(
            torch.from_numpy(partnet_pcd_in_fg_vox_grid).unsqueeze(0),
            normalize=True)
    partnet_pcd_norm = partnet_pcd_norm[0].to(torch.float32)

    pts_data_whole = Data(pos=partnet_pcd_norm,
                          part_label=torch.from_numpy(new_labels).to(torch.int64))

    random.seed(319)
    points, values = gather_hdf5.sample_points_values(fg_voxels, pt_sample_res)

    pv_indices = np.arange(len(points))
    np.random.seed(319)
    np.random.shuffle(pv_indices)
    points = points[pv_indices]
    values = values[pv_indices]

    fg_occ_points = points[values.astype('bool').flatten()]
    fg_occ_points_indices = np.where(values >= 0.5)[0]
    # print(fg_occ_points.shape)

    fg_closest_indices = ops.find_nearest_point(
        fg_occ_points,
        partnet_pcd_in_fg_vox_grid)
    fg_occupied_points_labels = new_labels[fg_closest_indices]

    fg_part_indices = []
    # find indices within occupied points that belong to separate parts
    for new_label in range(len(unique_names)):
        fg_part_indices.append(
            np.where(fg_occupied_points_labels == new_label)[0].tolist())

    if vis:
        colors = np.zeros((len(fg_occ_points), 3))
        for i in range(len(unique_names)):
            colors[fg_part_indices[i]] = all_colors[i]
        trimesh.points.PointCloud(fg_occ_points, colors).export(
            'data_prep/tmp/partitioned_occupied_points.ply')
    
    # find indices within all points that belong to separate parts
    for i, pi in enumerate(fg_part_indices):
        fg_part_indices[i] = fg_occ_points_indices[pi].tolist()

    if vis: 
        colors = np.zeros((len(points), 3))
        for i in range(len(unique_names)):
            colors[fg_part_indices[i]] = all_colors[i]
        trimesh.points.PointCloud(points, colors).export(
            'data_prep/tmp/partitioned_all_points.ply')

    part_num_indices, all_indices = convert_fg_part_indices_to_flat_list(
        fg_part_indices)
    points = torch.from_numpy(points).to('cuda', dtype=torch.float32)
    points = points.unsqueeze(0)
    normalized_points = kaolin.ops.pointcloud.center_points(points, normalize=True)
    normalized_points = normalized_points.cpu().numpy()
    values = values[None, :]

    return {
        'part_num_indices': [part_num_indices],     # 1, num_parts
        'all_indices': [all_indices],               # 1, variable length
        'normalized_points': normalized_points,     # 1, num_points, 3
        'values': values,                           # 1, num_points, 1
        'pts_data': pts_data,
        'pts_whole_data': pts_data_whole
    }


def get_info_from_voxels(anno_id, res, entire_mesh):
    """Given ShapeNet model_id and pt_sample_res,
    return model voxels and binvox transformation
    """
    obj_dir = os.path.join(partnet_dir, anno_id, 'vox_models')
    misc.check_dir(obj_dir)
    gt_mesh_path = os.path.join(obj_dir, f'{anno_id}.obj')
    entire_mesh.export(gt_mesh_path)
    vox_path = os.path.join(obj_dir, f'{anno_id}_{res}.binvox')
    vox_c_path = os.path.join(obj_dir, f'{anno_id}_{res}_c.binvox')
    if not os.path.exists(vox_c_path):
        ops.setup_vox(obj_dir)
        ops.voxelize_obj(
            obj_dir,
            f'{anno_id}.obj',
            res,
            vox_c_path,
            vox_path)
        ops.teardown_vox(obj_dir)
    voxels = ops.load_voxels(vox_c_path)
    binvox_xform = transform.get_transform_from_binvox_centered(
        vox_c_path, vox_path)
    return voxels, binvox_xform     


def convert_fg_part_indices_to_flat_list(fg_part_indices):
    part_num_indices = []
    all_indices = []
    for i in range(len(fg_part_indices)):
        part_indices = fg_part_indices[i]
        part_num_indices.append(len(part_indices))
        all_indices += part_indices
    return part_num_indices, all_indices


def convert_flat_list_to_fg_part_indices(part_num_indices, all_indices):
    fg_part_indices = []
    up_to_now = 0
    for i in range(len(part_num_indices)):
        curr_num_indices = part_num_indices[i]
        fg_part_indices.append(all_indices[up_to_now:up_to_now+curr_num_indices])
        up_to_now += curr_num_indices
    return fg_part_indices


if __name__ == "__main__":
    # pass one to create the preprocess_16 dataset
    # export_data(train_ids, save_data=False, start=0, end=4489)
    # export_data(train_ids, save_data=False, start=0, end=1554)
    # exit(0)
    
    # pass two to create the four_parts dataset
    good_indices = np.load('data/chair_am_four_parts_16_0_4489.npy')
    data_pt = 'data/Chair_train_new_ids_to_objs_16_0_4489.json'
    with open(data_pt, 'r') as f:
        data: Dict = json.load(f)
    all_ids = np.array(list(data.keys()))
    ids_w_four_parts = []
    for x in train_ids:
        if x['anno_id'] in all_ids:
            ids_w_four_parts.append(x)
    ids_w_four_parts = [ids_w_four_parts[x] for x in good_indices]

    # export_data(ids_w_four_parts, save_data=True,
    #             start=0, end=len(ids_w_four_parts))
    export_data(ids_w_four_parts, save_data=True,
                start=0, end=100)
    exit(0)

    # merge_partnet_after_merging('39446', info=True)
    # exit(0)

    unique_name_to_new_id, all_entire_meshes, all_ori_ids_to_new_ids,\
            all_obbs, all_name_to_obbs =\
                export_data(ids_w_four_parts, save_data=True, start=0, end=10)
    # np.savez_compressed("data_prep/tmp/data.npz",
    #                     all_entire_meshes=all_entire_meshes,
    #                     all_ori_ids_to_new_ids=all_ori_ids_to_new_ids,
    #                     all_obbs=all_obbs,
    #                     all_name_to_obbs=all_name_to_obbs)

    with open('data/Chair_train_new_ids_to_objs_8_0_100.json', 'r') as f:
        all_obs = json.load(f)
    keys = list(all_obs.keys())

    # with open(
    #     f'data/{cat_name}_part_name_to_new_id_8_{0}_{2000}.json',
    #     'r') as f:
    #     unique_name_to_new_id = json.load(f)

    # # anno_id = '43941'
    # # model_idx = 3
    # anno_id = '38725'
    # model_idx = 6
    model_idx = 0
    anno_id = keys[model_idx]
    print(anno_id)

    make_data_for_one(anno_id,
                      unique_name_to_new_id,
                      all_entire_meshes[model_idx],
                      all_ori_ids_to_new_ids[model_idx],
                      vis=True)