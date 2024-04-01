import numpy as np
from utils import binvox_rw
from scipy.spatial.transform import Rotation as R


def get_transform_from_binvox(path_c):
    """Outputs voxel to mesh transform
    """
    with open(path_c, 'r', encoding="cp437") as f:
        # Read first 4 lines of file containing transformation info
        header = [next(f).strip().split() for x in range(4)]

    assert header[1][0] == 'dim'
    assert header[2][0] == 'translate'
    assert header[3][0] == 'scale'
    assert len(header[1]) == 4
    assert len(header[2]) == 4
    assert len(header[3]) == 2

    try:
        dim = [int(header[1][i]) for i in range(1, 4)]
        translate = [float(header[2][i]) for i in range(1, 4)]
        scale = float(header[3][1])
    except ValueError:
        print(
            "Unexpected val type when parsing binvox transformation info")
        exit(-1)

    with open(path_c, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
        model = model.data.astype(int)
    transform = {
        'dim': dim,
        'translate': translate,
        'scale': scale
    }

    return transform


def voxel_space_to_mesh_space(transform, points_in_voxel_grid):
    """NOTE: need to check if it's fine to subtract 0.5
    """
    dim = transform['dim']
    translate = transform['translate']
    scale = transform['scale']

    points_t = np.transpose(points_in_voxel_grid)
    voxel_grid_x = points_t[0]
    voxel_grid_y = points_t[1]
    voxel_grid_z = points_t[2]

    original_x = (voxel_grid_x - 0.5) * (scale / dim[0]) + translate[0]
    original_y = (voxel_grid_y - 0.5) * (scale / dim[1]) + translate[1]
    original_z = (voxel_grid_z - 0.5) * (scale / dim[2]) + translate[2]

    original_points = np.stack((original_x, original_y, original_z), axis=0)
    return np.squeeze(np.transpose(original_points))


def mesh_space_to_voxel_space(transform, points_in_mesh_space):
    dim = transform['dim']
    translate = transform['translate']
    scale = transform['scale']

    xs = points_in_mesh_space[:, 0:1]
    ys = points_in_mesh_space[:, 1:2]
    zs = points_in_mesh_space[:, 2:3]

    # i = (xs - translate[0]) * (dim[0] / scale) + 0.5
    # j = (ys - translate[1]) * (dim[1] / scale) + 0.5
    # k = (zs - translate[2]) * (dim[2] / scale) + 0.5
    i = (xs - translate[0]) * (dim[0] / scale)
    j = (ys - translate[1]) * (dim[1] / scale)
    k = (zs - translate[2]) * (dim[2] / scale)
    new_points = np.concatenate((i, j, k), axis=1)

    return new_points


def get_mids(verts):
    min_x = np.min(verts[:, 0])
    max_x = np.max(verts[:, 0])
    min_y = np.min(verts[:, 1])
    max_y = np.max(verts[:, 1])
    min_z = np.min(verts[:, 2])
    max_z = np.max(verts[:, 2])
    mid_x = (max_x + min_x) / 2
    mid_y = (max_y + min_y) / 2
    mid_z = (max_z + min_z) / 2

    return mid_x, mid_y, mid_z


# def get_mids(verts):
#     return np.mean(verts[:, 0]), np.mean(verts[:, 1]), np.mean(verts[:, 2])


def get_unnorm_to_center_t(model, model_unnorm):
    verts = np.argwhere(model > 0).astype(np.float32, copy=False)
    verts_unnorm = np.argwhere(model_unnorm > 0).astype(np.float32, copy=False)

    o_mid_x, o_mid_y, o_mid_z = get_mids(verts)
    mid_x, mid_y, mid_z = get_mids(verts_unnorm)

    return [o_mid_x-mid_x, o_mid_y-mid_y, o_mid_z-mid_z]


def get_transform_from_binvox_centered(path_c, path_nc):
    # gives voxel to mesh transform
    with open(path_c, 'r', encoding="cp437") as f:
        header = [next(f).strip().split() for x in range(4)]

    assert header[1][0] == 'dim'
    assert header[2][0] == 'translate'
    assert header[3][0] == 'scale'
    assert len(header[1]) == 4
    assert len(header[2]) == 4
    assert len(header[3]) == 2

    try:
        dim = [int(header[1][i]) for i in range(1, 4)]
        translate = [float(header[2][i]) for i in range(1, 4)]
        scale = float(header[3][1])
    except ValueError:
        print(
            "Unexpected val type when parsing binvox transformation info")
        exit(-1)

    with open(path_c, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
        model = model.data.astype(int)
    with open(path_nc, 'rb') as f:
        model_unnorm = binvox_rw.read_as_3d_array(f)
        model_unnorm = model_unnorm.data.astype(int)
    to_center = get_unnorm_to_center_t(model, model_unnorm)
    transform = {
        'dim': dim,
        'translate': translate,
        'scale': scale,
        'to_center': to_center
    }

    return transform


def voxel_space_to_mesh_space_centered(transform, points_in_voxel_grid):
    """NOTE: need to check if it's fine to subtract 0.5
    """
    dim = transform['dim']
    translate = transform['translate']
    scale = transform['scale']

    points_t = np.transpose(points_in_voxel_grid)
    voxel_grid_x = points_t[0]
    voxel_grid_y = points_t[1]
    voxel_grid_z = points_t[2]

    to_center = transform['to_center']
    voxel_grid_x -= to_center[0]
    voxel_grid_y -= to_center[1]
    voxel_grid_z -= to_center[2]

    # original_x = (voxel_grid_x - 0.5) * (scale / dim[0]) + translate[0]
    # original_y = (voxel_grid_y - 0.5) * (scale / dim[1]) + translate[1]
    # original_z = (voxel_grid_z - 0.5) * (scale / dim[2]) + translate[2]
    original_x = (voxel_grid_x) * (scale / dim[0]) + translate[0]
    original_y = (voxel_grid_y) * (scale / dim[1]) + translate[1]
    original_z = (voxel_grid_z) * (scale / dim[2]) + translate[2]

    original_points = np.stack((original_x, original_y, original_z), axis=0)
    return np.squeeze(np.transpose(original_points))


def mesh_space_to_voxel_space_centered(transform, points_in_mesh_space):
    dim = transform['dim']
    translate = transform['translate']
    scale = transform['scale']
    xs = points_in_mesh_space[:, 0:1]
    ys = points_in_mesh_space[:, 1:2]
    zs = points_in_mesh_space[:, 2:3]

    # i = (xs - translate[0]) * (dim[0] / scale) + 0.5
    # j = (ys - translate[1]) * (dim[1] / scale) + 0.5
    # k = (zs - translate[2]) * (dim[2] / scale) + 0.5
    i = (xs - translate[0]) * (dim[0] / scale)
    j = (ys - translate[1]) * (dim[1] / scale)
    k = (zs - translate[2]) * (dim[2] / scale)

    to_center = transform['to_center']
    i += to_center[0]
    j += to_center[1]
    k += to_center[2]

    new_points = np.concatenate((i, j, k), axis=1)
    return new_points


def transform_points(points, transform, ones_or_zeros='ones'):
    assert ones_or_zeros in ['ones', 'zeros']
    num_points = points.shape[0]
    if ones_or_zeros == 'ones':
        padding = np.ones((num_points, 1), dtype=np.float32)
    else:
        padding = np.zeros((num_points, 1), dtype=np.float32)
    h_points = np.transpose(np.concatenate((points, padding), axis=-1))
    xformed_points = np.transpose(np.matmul(transform, h_points))[:, :3]
    return xformed_points


def transform_one_point(point, transform, ones_or_zeros='ones'):
    """
    in shape: (3, )
    out shape: (3, )
    """
    return transform_points(point[None, :], transform, ones_or_zeros)[0]