import os
import json
import torch
from anytree import AnyNode
import subprocess
import numpy as np
import scipy.ndimage as ndimage
from utils import misc, binvox_rw, transform


def load_voxels(vox_path):
    with open(vox_path, 'rb') as f:
        voxel_model_dense = binvox_rw.read_as_3d_array(f)
        voxels = voxel_model_dense.data.astype(int)
    return voxels


def setup_vox(obj_dir):
    subprocess.call(
        ["cp utils/binvox "+obj_dir+"/binvox"], shell=True)
    subprocess.call(
        ["cp utils/voxelize.sh "+obj_dir+"/voxelize.sh"], shell=True)


def teardown_vox(obj_dir):
    subprocess.call(["rm {}/voxelize.sh".format(obj_dir)], shell=True)
    subprocess.call(["rm {}/binvox".format(obj_dir)], shell=True)
    subprocess.call(["killall Xvfb"], shell=True)


def voxelize_obj(obj_dir, obj_filename, res,
                 out_vox_c_path, out_vox_nc_path,
                 bbox="", min_max=["", "", "", "", "", ""]):
    assert os.path.exists(os.path.join(obj_dir, 'binvox'))
    assert os.path.exists(os.path.join(obj_dir, 'voxelize.sh'))
    files = misc.sorted_alphanumeric(os.listdir(obj_dir))
    files.remove('voxelize.sh')
    files.remove('binvox')
    obj_id = os.path.splitext(obj_filename)[0]
    devnull = open(os.devnull, 'w')
    if bbox == "":
        call_string_c = "cd {} && bash voxelize.sh {} {} c".format(
            obj_dir, obj_filename, res)
    else:
        call_string_c =\
            "cd {} && bash voxelize.sh {} {} c {} {} {} {} {} {} {}".format(
                obj_dir, obj_filename, res, bbox,
                min_max[0], min_max[1], min_max[2],
                min_max[3], min_max[4], min_max[5])
    mv_c = "mv {}/{}.binvox {}".format(
        obj_dir, obj_id, out_vox_c_path)
    subprocess.call([call_string_c], shell=True,
        stdout=devnull, stderr=devnull
    )
    subprocess.call([mv_c], shell=True)
    if bbox == "":
        call_string_nc = "cd {} && bash voxelize.sh {} {} nc".format(
            obj_dir, obj_filename, res)
    else:
        call_string_nc =\
            "cd {} && bash voxelize.sh {} {} nc {} {} {} {} {} {} {}".format(
                obj_dir, obj_filename, res, bbox,
                min_max[0], min_max[1], min_max[2],
                min_max[3], min_max[4], min_max[5])
    mv_nc = "mv {}/{}.binvox {}".format(
        obj_dir, obj_id, out_vox_nc_path)
    subprocess.call([call_string_nc], shell=True,
        stdout=devnull, stderr=devnull
    )
    subprocess.call([mv_nc], shell=True)


def parse_labels_txt(in_path):
    """Returns partnet labels, unique labels, remapped labels,
    and unique label to remapped label map
    """
    with open(in_path) as f:
        lines = f.readlines()
    labels = np.array([int(x.strip()) for x in lines])
    unique_labels = np.unique(labels)
    short = np.arange(len(unique_labels))   # 0, 1, 2, 3 ...
    labels_map = {}
    for l in unique_labels:
        labels_map[l] = short[unique_labels.tolist().index(l)]
    remapped_labels = np.array([labels_map[x] for x in labels])
    return labels, unique_labels, remapped_labels, labels_map


def get_partnet_labels_and_obj_names(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    labels_to_obj_names = {}

    def traverse(node):
        if 'children' in node and node['children']:
            for child in node['children']:
                traverse(child)
        else:
            # node does not have more children
            # we are at the leaf
            # NOTE: a leaf node could have multiple mesh parts!!!

            # # [DEPRECATED] this only considers one part per leaf node
            # assert len(node['objs']) == 1, f"{node['objs']}"
            # labels_to_obj_names[node['ori_id']] = node['objs'][0]

            # [UPDATED] this considers multiple parts per peaf node
            # labels_to_obj_names[node['ori_id']] = node['objs']  # result_after_merging
            labels_to_obj_names[node['id']] = node['objs']      # result

    for node in data:
        traverse(node)

    return labels_to_obj_names


def find_nearest_point(source, target):
    """Given source and target points, return indices of target points that
    are closest to each source point
    source: n, 3
    target: m, 3
    out: n
    """
    # Reshape source points to (n, 1, 3) and repeat target points to (1, m, 3)
    source_expanded = source[:, None, :]
    target_repeated = target[None, :, :]

    # Compute squared Euclidean distances (using broadcasting)
    distances_squared = np.sum((source_expanded - target_repeated) ** 2, axis=2)

    # Find indices of the closest target points
    closest_point_indices = np.argmin(distances_squared, axis=1)

    return closest_point_indices