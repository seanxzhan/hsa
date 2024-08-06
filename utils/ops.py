import os
import json
import torch
import kaolin
import trimesh
import subprocess
import numpy as np
import scipy.ndimage as ndimage
from utils import binvox_rw


def export_mesh_norm(vertices, faces, path):
    """vertices, faces: torch.Tensor
    """
    vertices = kaolin.ops.pointcloud.center_points(
        vertices.unsqueeze(0), normalize=True).squeeze(0)
    vertices = vertices.numpy()
    faces = faces.numpy()
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.export(path)
    return mesh


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
    # subprocess.call(["killall Xvfb"], shell=True)


def voxelize_obj(obj_dir, obj_filename, res,
                 out_vox_c_path, out_vox_nc_path,
                 display=99,
                 bbox="", min_max=["", "", "", "", "", ""]):
    assert os.path.exists(os.path.join(obj_dir, 'binvox'))
    assert os.path.exists(os.path.join(obj_dir, 'voxelize.sh'))
    # files = misc.sorted_alphanumeric(os.listdir(obj_dir))
    # files.remove('voxelize.sh')
    # files.remove('binvox')
    obj_id = os.path.splitext(obj_filename)[0]
    devnull = open(os.devnull, 'w')
    if bbox == "":
        call_string_c = "cd {} && bash voxelize.sh {} {} c {}".format(
            obj_dir, obj_filename, res, display)
    else:
        call_string_c =\
            "cd {} && bash voxelize.sh {} {} c {} {} {} {} {} {} {} {}".format(
                obj_dir, obj_filename, res, display, bbox,
                min_max[0], min_max[1], min_max[2],
                min_max[3], min_max[4], min_max[5])
    mv_c = "mv {}/{}.binvox {}".format(
        obj_dir, obj_id, out_vox_c_path)
    subprocess.call([call_string_c], shell=True,
        stdout=devnull,
        stderr=devnull
    )
    subprocess.call([mv_c], shell=True)
    if bbox == "":
        call_string_nc = "cd {} && bash voxelize.sh {} {} nc {}".format(
            obj_dir, obj_filename, res, display)
    else:
        call_string_nc =\
            "cd {} && bash voxelize.sh {} {} nc {} {} {} {} {} {} {} {}".format(
                obj_dir, obj_filename, res, display, bbox,
                min_max[0], min_max[1], min_max[2],
                min_max[3], min_max[4], min_max[5])
    mv_nc = "mv {}/{}.binvox {}".format(
        obj_dir, obj_id, out_vox_nc_path)
    subprocess.call([call_string_nc], shell=True,
        stdout=devnull,
        stderr=devnull
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


import torch.nn.functional as F
def bin2sdf_torch(input):
    input = input.float()
    device = input.device
    fill_map = torch.zeros_like(input, dtype=torch.bool)
    output = torch.zeros_like(input, dtype=torch.float32)
    
    # Structuring element for 3D convolution
    struct_elem = torch.ones((1, 1, 3, 3, 3), device=device)

    # Fill inside
    changing_map = input.clone()
    sdf_in = -1
    while torch.sum(fill_map) != torch.sum(input):
        changing_map_padded = F.pad(changing_map, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        changing_map_new = F.conv3d(changing_map_padded.unsqueeze(0).unsqueeze(0), struct_elem, stride=1).squeeze() == 27
        changing_map_new = changing_map_new.float()
        fill_map = fill_map | (changing_map_new != changing_map)
        output[fill_map & (changing_map_new != changing_map)] = sdf_in
        changing_map = changing_map_new.clone()
        sdf_in -= 1

    # Fill outside
    changing_map = input.clone()
    sdf_out = 1
    while torch.sum(fill_map) != torch.numel(input):
        changing_map_padded = F.pad(changing_map, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        changing_map_new = F.conv3d(changing_map_padded.unsqueeze(0).unsqueeze(0), struct_elem, stride=1).squeeze() > 0
        changing_map_new = changing_map_new.float()
        fill_map = fill_map | (changing_map_new != changing_map)
        output[fill_map & (changing_map_new != changing_map)] = sdf_out
        changing_map = changing_map_new.clone()
        sdf_out += 1

    # Normalization
    output[output < 0] /= (-sdf_in - 1)
    output[output > 0] /= (sdf_out - 1)

    return output


def bin2sdf_torch_1(input):
    input = input.float()
    device = input.device
    
    # Initialize distance transform
    max_dist = torch.tensor(input.shape[1:], dtype=torch.float32, device=device).norm().item()
    
    # Inside distance
    dist_inside = torch.where(input == 1, 0.0, max_dist)
    for _ in range(int(max_dist)):
        dist_inside_padded = F.pad(dist_inside, (1, 1, 1, 1, 1, 1), mode='constant', value=max_dist)
        dist_inside_new = F.conv3d(dist_inside_padded.unsqueeze(0).unsqueeze(0), 
                                   torch.ones((1, 1, 3, 3, 3), device=device), stride=1).squeeze()
        dist_inside = torch.min(dist_inside, dist_inside_new)
        dist_inside[input == 1] = 0.0
    
    # Outside distance
    dist_outside = torch.where(input == 0, 0.0, max_dist)
    for _ in range(int(max_dist)):
        dist_outside_padded = F.pad(dist_outside, (1, 1, 1, 1, 1, 1), mode='constant', value=max_dist)
        dist_outside_new = F.conv3d(dist_outside_padded.unsqueeze(0).unsqueeze(0), 
                                    torch.ones((1, 1, 3, 3, 3), device=device), stride=1).squeeze()
        dist_outside = torch.min(dist_outside, dist_outside_new)
        dist_outside[input == 0] = 0.0
    
    # Combine inside and outside distances
    output = -dist_inside + dist_outside

    # Normalization
    output = output / max_dist

    return output


def bin2sdf_torch_2(input):
    input = input.float()
    device = input.device
    
    # Initialize distance transform
    max_dist = torch.tensor(input.shape[1:], dtype=torch.float32, device=device).norm().item()
    
    # Inside distance
    dist_inside = torch.where(input == 1, 0.0, max_dist)
    # for _ in range(int(max_dist)):
    dist_inside_padded = F.pad(dist_inside, (1, 1, 1, 1, 1, 1), mode='constant', value=max_dist)
    dist_inside_new = F.conv3d(dist_inside_padded.unsqueeze(0).unsqueeze(0), 
                                torch.ones((1, 1, 3, 3, 3), device=device), stride=1).squeeze()
    dist_inside = torch.min(dist_inside, dist_inside_new)
    dist_inside[input == 1] = 0.0
    
    # Outside distance
    dist_outside = torch.where(input == 0, 0.0, max_dist)
    # for _ in range(int(max_dist)):
    dist_outside_padded = F.pad(dist_outside, (1, 1, 1, 1, 1, 1), mode='constant', value=max_dist)
    dist_outside_new = F.conv3d(dist_outside_padded.unsqueeze(0).unsqueeze(0), 
                                torch.ones((1, 1, 3, 3, 3), device=device), stride=1).squeeze()
    dist_outside = torch.min(dist_outside, dist_outside_new)
    dist_outside[input == 0] = 0.0
    
    # Combine inside and outside distances
    output = -dist_inside + dist_outside

    # Normalization
    output = output / max_dist

    return output


def bin2sdf_torch_3(input):
    return -2 * input + 1


def bin2sdf_torch_4(input):
    input = input.float()
    device = input.device
    
    # Initialize distance transform
    max_dist = torch.tensor(input.shape[1:], dtype=torch.float32, device=device).norm().item()
    # print(max_dist)
    # exit(0)
    # Inside distance
    # dist_inside = torch.where(input == 1, 0.0, max_dist)
    dist_inside = (1 - input) * max_dist
    # for _ in range(int(max_dist)):
    dist_inside_padded = F.pad(dist_inside, (1, 1, 1, 1, 1, 1), mode='constant', value=max_dist)
    dist_inside_new = F.conv3d(dist_inside_padded.unsqueeze(0).unsqueeze(0), 
                                torch.ones((1, 1, 3, 3, 3), device=device), stride=1).squeeze()
    dist_inside = torch.min(dist_inside, dist_inside_new)
    # dist_inside[input == 1] = 0.0
    
    # Outside distance
    # dist_outside = torch.where(input == 0, 0.0, max_dist)
    dist_outside = input * max_dist
    # for _ in range(int(max_dist)):
    dist_outside_padded = F.pad(dist_outside, (1, 1, 1, 1, 1, 1), mode='constant', value=max_dist)
    dist_outside_new = F.conv3d(dist_outside_padded.unsqueeze(0).unsqueeze(0), 
                                torch.ones((1, 1, 3, 3, 3), device=device), stride=1).squeeze()
    dist_outside = torch.min(dist_outside, dist_outside_new)
    # dist_outside[input == 0] = 0.0
    
    # Combine inside and outside distances
    output = -dist_inside + dist_outside

    # Normalization
    output = output / max_dist

    return output


def bin2sdf_torch_5(input_tensor, epsilon=1e-6):
    device = input_tensor.device
    # Create meshgrid for 3D coordinates
    D, H, W = input_tensor.shape
    z = torch.arange(0, D, device=device)
    y = torch.arange(0, H, device=device)
    x = torch.arange(0, W, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    
    # Stack coordinates into a single tensor
    coords = torch.stack([zz, yy, xx], dim=-1).float()
    
    # Create a mask for foreground and background
    foreground_mask = input_tensor == 1
    
    # Get foreground coordinates
    foreground_coords = coords[foreground_mask].view(-1, 3)
    
    # Compute squared Euclidean distance to each foreground pixel
    distances = torch.cdist(coords.view(-1, 3).float(), foreground_coords.float(), p=2.0)
    
    # Take the minimum distance for each background pixel
    min_distances, _ = torch.min(distances, dim=1)
    min_distances = min_distances.view(D, H, W)
    
    # Apply differentiable approximation (soft minimum) if needed
    min_distances = torch.sqrt(min_distances + epsilon)
    
    return min_distances


def bin2sdf_torch_6(input):
    return -input + 1


def bin2sdf_torch_7(input):
    return -smoothing_sinh(input)+1


def smoothing_sinh(input):
    return 0.1 * torch.sinh(4.625*(input-0.5)) + 0.5


def smoothing_sinh_np(input):
    return 0.1 * np.sinh(4.625*(input-0.5)) + 0.5
    # return 0.01 * np.sinh(9.21*(input-0.5)) + 0.5
    # return 0.05 * np.sinh(5.996*(input-0.5)) + 0.5
    # return 0.075 * np.sinh(5.19169*(input-0.5)) + 0.5


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


def vectorized_trilinear_interpolate(volume, points,
                                     vox_res):
    """
    Performs vectorized trilinear interpolation of a 3D volume at specified points.

    :param volume: Input volume to interpolate, shape (B, C, D, H, W)
    :param points: Points at which to interpolate, shape (B, N, 3)
                    Points are assumed to be in the coordinate frame of the volume
                    i.e., in the range [0, D-1] x [0, H-1] x [0, W-1].
    :return: Interpolated values, shape (B, C, N)
    """
    B, C, D, H, W = volume.shape

    clamped = points.clamp(0, vox_res-1)
    points_int = clamped.long()
    points_frac = clamped - points_int

    # Calculate the indices of the 8 neighboring points
    x0 = points_int[..., 0]
    y0 = points_int[..., 1]
    z0 = points_int[..., 2]
    x1 = (x0 + 1).clamp(max=W-1)
    y1 = (y0 + 1).clamp(max=H-1)
    z1 = (z0 + 1).clamp(max=D-1)

    # Create a grid of indices for batch and channel dimensions
    batch_indices = torch.arange(
        B, dtype=torch.long, device=volume.device).view(B, 1).expand(
            B, points.size(0))
    channel_indices = torch.arange(
        C, dtype=torch.long, device=volume.device).view(1, C, 1).expand(
            B, C, points.size(0))

    # Retrieve values from the volume at the corner points
    v000 = volume[batch_indices, channel_indices, z0, y0, x0]
    v100 = volume[batch_indices, channel_indices, z0, y0, x1]
    v010 = volume[batch_indices, channel_indices, z0, y1, x0]
    v110 = volume[batch_indices, channel_indices, z0, y1, x1]
    v001 = volume[batch_indices, channel_indices, z1, y0, x0]
    v101 = volume[batch_indices, channel_indices, z1, y0, x1]
    v011 = volume[batch_indices, channel_indices, z1, y1, x0]
    v111 = volume[batch_indices, channel_indices, z1, y1, x1]

    # Interpolate along the x-axis
    vx00 = (1 - points_frac[..., 0]) * v000 + points_frac[..., 0] * v100
    vx10 = (1 - points_frac[..., 0]) * v010 + points_frac[..., 0] * v110
    vx01 = (1 - points_frac[..., 0]) * v001 + points_frac[..., 0] * v101
    vx11 = (1 - points_frac[..., 0]) * v011 + points_frac[..., 0] * v111

    # Interpolate along the y-axis
    vxy0 = (1 - points_frac[..., 1]) * vx00 + points_frac[..., 1] * vx10
    vxy1 = (1 - points_frac[..., 1]) * vx01 + points_frac[..., 1] * vx11

    # Interpolate along the z-axis
    interpolated = (1 - points_frac[..., 2]) * vxy0 + points_frac[..., 2] * vxy1

    return interpolated
