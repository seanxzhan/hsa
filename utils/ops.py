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


def bin2sdf(vox_path='', input=None):
    """Converts voxels into SDF grid field.
       Negative for inside, positive for outside. Range: (-1, 1)

    Args:
        input (np.ndarray): voxels of shape (dim, dim, dim)

    Returns:
        np.ndarray: SDF representation of shape (dim, dim, dim)
    """
    assert (vox_path != '') != (input is not None),\
        "must supply either [vox_path] or [input]"

    if vox_path != '':
        with open(vox_path, 'rb') as f:
            voxel_model_dense = binvox_rw.read_as_3d_array(f)
            input = voxel_model_dense.data.astype(int)

    fill_map = np.zeros(input.shape, dtype=np.bool_)
    output = np.zeros(input.shape, dtype=np.float16)
    # fill inside
    changing_map = input.copy()
    sdf_in = -1
    while np.sum(fill_map) != np.sum(input):
        changing_map_new = ndimage.binary_erosion(changing_map)
        fill_map[np.where(changing_map_new!=changing_map)] = True
        output[np.where(changing_map_new!=changing_map)] = sdf_in
        changing_map = changing_map_new.copy()
        sdf_in -= 1
    # fill outside.
    # No need to fill all of them, since during training,
    # outside part will be masked.
    changing_map = input.copy()
    sdf_out = 1
    while np.sum(fill_map) != np.size(input):
        changing_map_new = ndimage.binary_dilation(changing_map)
        fill_map[np.where(changing_map_new!=changing_map)] = True
        output[np.where(changing_map_new!=changing_map)] = sdf_out
        changing_map = changing_map_new.copy()
        sdf_out += 1
        # if sdf_out == -sdf_in:
        #     break
    # Normalization
    output[np.where(output < 0)] /= (-sdf_in-1)
    output[np.where(output > 0)] /= (sdf_out-1)

    output = output.astype(np.float32)
    return output


def sample_near_sdf_surface(sdf_grid, voxel_grid,
                            use_sdf_values=False):
    # print(np.sum(sdf_grid == 0))
    scale = 5  # Control the sensitivity near the surface
    surface_weights = np.exp(-scale * np.abs(sdf_grid))

    inside_mask = sdf_grid <= 0
    outside_mask = sdf_grid > 0
    num_negative = inside_mask.sum()
    num_positive = outside_mask.sum()
    surface_weights[inside_mask] *= (num_positive / num_negative)
    # surface_weights[outside_mask] *= (num_negative / num_positive)

    surface_weights /= surface_weights.sum()

    flat_grid = np.indices(sdf_grid.shape).reshape(3, -1).T
    flat_weights = surface_weights.reshape(-1,)

    num_samples = 10000  # Number of points to sample
    chosen_indices = np.random.choice(flat_grid.shape[0],
                                      size=num_samples,
                                      p=flat_weights,
                                      replace=False)
    sampled_points = flat_grid[chosen_indices]

    if not use_sdf_values:
        flat_voxel_values = voxel_grid.reshape(-1,)
        sampled_values = flat_voxel_values[chosen_indices]
    else:
        flat_sdf_values = sdf_grid.reshape(-1,)
        sampled_values = flat_sdf_values[chosen_indices]

    # import matplotlib.pyplot as plt
    # plt.hist(sampled_values, bins=50, color='blue')
    # plt.title("Histogram of Sampled SDF Values")
    # plt.xlabel("SDF Value")
    # plt.ylabel("Frequency")
    # plt.xlim(-1, 1)
    # misc.save_fig(plt, '', 'data_prep/tmp/partitioned_hist.png')
    # plt.close()

    return sampled_points, sampled_values