import torch
import kaolin
import mcubes
import trimesh
import numpy as np
from utils import transform


def reconstruct_from_grid(sdf,
                          save_mesh=True, mesh_out_path=None, 
                          xform=True, transform_info={},
                          negate=True, level_set=0):
    # marching_cubes thinks positive means inside, negative means outside
    sdf_grid = -sdf if negate else sdf
    vertices, triangles = mcubes.marching_cubes(sdf_grid, level_set)
    if xform:
        assert transform_info != {}
        vertices = transform.voxel_space_to_mesh_space_centered(
            transform_info, vertices)
    # vertices[:, 0] = vertices[:, 0] / (128 - 1) - 0.5
    # vertices[:, 1] = vertices[:, 1] / (128 - 1) - 0.5
    # vertices[:, 2] = vertices[:, 2] / (128 - 1) - 0.5
    mesh = trimesh.Trimesh(vertices, triangles)
    if save_mesh:
        trimesh.exchange.export.export_mesh(
            mesh, mesh_out_path, file_type='obj')
    return mesh


def make_query_points(vox_res, limits=[(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]):
    xx, yy, zz = make_grid_around_mesh(vox_res, limits)
    xx = np.reshape(xx, (vox_res**3, 1))
    yy = np.reshape(yy, (vox_res**3, 1))
    zz = np.reshape(zz, (vox_res**3, 1))
    query_points = np.concatenate([xx, yy, zz], axis=-1)
    return query_points
    query_points = torch.from_numpy(query_points).to(torch.float32)
    query_points = query_points.unsqueeze(0).to(device=kwargs['device'])
    query_points = kaolin.ops.pointcloud.center_points(query_points, normalize=True)
    values = model(query_points, **kwargs).squeeze(0).cpu().numpy()
    sdf_grid = np.reshape(values, (vox_res, vox_res, vox_res))
    return reconstruct_from_grid(sdf_grid, **kwargs)


def make_grid_around_mesh(res, limits=[(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]):
# def make_grid_around_mesh(res, limits=[(-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6)]):
    """Makes grid points around the mesh.
    These grid points are in the mesh's world space.
    """
    x_min = limits[0][0]
    x_max = limits[0][1]
    y_min = limits[1][0]
    y_max = limits[1][1]
    z_min = limits[2][0]
    z_max = limits[2][1]
    w = x_max - x_min
    # X = np.linspace(x_min, x_max, res+1)
    # Y = np.linspace(y_min, y_max, res+1)
    # Z = np.linspace(z_min, z_max, res+1)
    X = np.linspace(x_min, x_max, res)
    Y = np.linspace(y_min, y_max, res)
    Z = np.linspace(z_min, z_max, res)
    xx, yy, zz = np.meshgrid(X, Y, Z)
    # xx = xx[:-1, :-1, :-1]
    # yy = yy[:-1, :-1, :-1]
    # zz = zz[:-1, :-1, :-1]
    # unit = w / res
    # adj = unit / 2
    # xx += adj
    # yy += adj
    # zz += adj
    return xx, yy, zz


