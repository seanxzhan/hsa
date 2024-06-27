import torch
import kaolin


def laplace_regularizer_const(mesh_verts, mesh_faces):
    term = torch.zeros_like(mesh_verts)
    norm = torch.zeros_like(mesh_verts[..., 0:1])

    v0 = mesh_verts[mesh_faces[:, 0], :]
    v1 = mesh_verts[mesh_faces[:, 1], :]
    v2 = mesh_verts[mesh_faces[:, 2], :]

    term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, mesh_faces[:, 0:1], two)
    norm.scatter_add_(0, mesh_faces[:, 1:2], two)
    norm.scatter_add_(0, mesh_faces[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term**2)

def loss_f(iterations, mesh_verts, mesh_faces, points, it):
    laplacian_weight = 0.1
    # iterations = 30000

    pred_points = kaolin.ops.mesh.sample_points(
        mesh_verts.unsqueeze(0), mesh_faces, 10000)[0][0]
    chamfer = kaolin.metrics.pointcloud.chamfer_distance(
        pred_points.unsqueeze(0), points.unsqueeze(0)).mean()
    if it > iterations//2:
        lap = laplace_regularizer_const(mesh_verts, mesh_faces)
        return chamfer + lap * laplacian_weight
    return chamfer