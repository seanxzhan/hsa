# NOTE: using flexicubes loss to learn occupancy network, overfit to an entire shape

import os
import json
import math
import h5py
import torch
import kaolin
import trimesh
import argparse
import numpy as np
from typing import Dict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from flexi.flexicubes import FlexiCubes
from flexi import render, util
from occ_networks.occflexi_network_2 import SmallMLPs
from utils import misc, ops, reconstruct, polyvis

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str)
parser.add_argument('--train', action="store_true")
parser.add_argument('--it', type=int)
parser.add_argument('--test', action="store_true")
parser.add_argument('--test_idx', '--ti', type=int)
args = parser.parse_args()

# ------------ hyper params ------------
name_to_cat = {
    'Chair': '03001627',
}
cat_name = 'Chair'
pt_sample_res = 64
occ_res = int(pt_sample_res / 2)
ds_start, ds_end = 0, 10
device = 'cuda'
lr = 0.001
iterations = 3000; iterations += 1
train_res = [512, 512]
fc_res = 31
num_shapes = 1
batch_size = 1
embed_dim = 128
dataset_id = 19
expt_id = 5
model_idx = 0

# ------------ data dirs ------------
partnet_dir = '/datasets/PartNet'
partnet_index_path = '/sota/partnet_dataset/stats/all_valid_anno_info.txt'
train_new_ids_to_objs_path = \
    f'data/{cat_name}_train_new_ids_to_objs_{dataset_id}_{ds_start}_{ds_end}.json'
with open(train_new_ids_to_objs_path, 'r') as f:
    train_new_ids_to_objs: Dict = json.load(f)
model_idx_to_anno_id = {}
for mi, ai in enumerate(train_new_ids_to_objs.keys()):
    model_idx_to_anno_id[mi] = ai
anno_id = model_idx_to_anno_id[model_idx]
print("anno_id: ", anno_id)

# ------------ logging ------------
results_dir = os.path.join('results', 'testflexi-occflexi',)
timelapse_dir = os.path.join(results_dir, 'training_timelapse')
print("timelapse dir: ", timelapse_dir)
timelapse = kaolin.visualize.Timelapse(timelapse_dir)
logs_path = os.path.join('logs', 'flexi', f'flexi_{expt_id}')
ckpt_dir = os.path.join(logs_path, 'ckpt'); misc.check_dir(ckpt_dir)
writer = SummaryWriter(os.path.join(logs_path, 'summary'))

# ------------ data loading ------------
train_data_path = \
    f'data/{cat_name}_train_{pt_sample_res}_{dataset_id}_{ds_start}_{ds_end}.hdf5'
train_data = h5py.File(train_data_path, 'r')
part_num_indices = train_data['part_num_indices']
all_indices = train_data['all_indices']
normalized_points = train_data['normalized_points'] # sampled points, shuffled
values = train_data['values']                       # vals of sampled points
occ = train_data['occ']                             # occ vals of flexi verts
part_nodes = train_data['part_nodes']
xforms = train_data['xforms']
extents = train_data['extents']
node_features = train_data['node_features']
adj = train_data['adj']
part_nodes = train_data['part_nodes']
relations = train_data['relations']
def my_load_mesh(model_idx):
    anno_id = model_idx_to_anno_id[model_idx]
    obj_dir = os.path.join(partnet_dir, anno_id, 'vox_models')
    assert os.path.exists(obj_dir)
    gt_mesh_path = os.path.join(obj_dir, f'{anno_id}.obj')
    gt_mesh = trimesh.load_mesh(gt_mesh_path)
    return util.load_mesh_vf_kaolin(gt_mesh.vertices, gt_mesh.faces, device)
def my_load_mesh_part(model_idx, part_verts, part_faces, part):
    verts = np.array(part_verts[part][model_idx]).reshape(-1, 3)
    faces = np.array(part_faces[part][model_idx]).reshape(-1, 3)
    return util.load_mesh_vf_kaolin(verts, faces, device)

# ------------ init flexicubes ------------
fc = FlexiCubes(device)
x_nx3, cube_fx8 = fc.construct_voxel_grid(fc_res)
x_nx3 *= 1.1
flexi_verts = x_nx3.to(device).unsqueeze(0)
def get_center_boundary_index(grid_res, device):
    v = torch.zeros((grid_res + 1, grid_res + 1, grid_res + 1),
                    dtype=torch.bool, device=device)
    v[grid_res // 2 + 1, grid_res // 2 + 1, grid_res // 2 + 1] = True
    center_indices = torch.nonzero(v.reshape(-1))
    v[grid_res // 2 + 1, grid_res // 2 + 1, grid_res // 2 + 1] = False
    v[:2, ...] = True
    v[-2:, ...] = True
    v[:, :2, ...] = True
    v[:, -2:, ...] = True
    v[:, :, :2] = True
    v[:, :, -2:] = True
    boundary_indices = torch.nonzero(v.reshape(-1))
    return center_indices, boundary_indices
center_indices, boundary_indices = get_center_boundary_index(fc_res, device)

# print(flexi_verts.shape)
# exit(0)

# flexi_verts = flexi_verts[0]
# print(torch.min(flexi_verts[:, 0]), torch.max(flexi_verts[:, 0]))
# print(torch.min(flexi_verts[:, 1]), torch.max(flexi_verts[:, 1]))
# print(torch.min(flexi_verts[:, 2]), torch.max(flexi_verts[:, 2]))

# print(np.min(normalized_points[:, 0]), np.max(normalized_points[:, 0]))
# print(np.min(normalized_points[:, 1]), np.max(normalized_points[:, 1]))
# print(np.min(normalized_points[:, 2]), np.max(normalized_points[:, 2]))
# exit(0)

# ------------ init gt meshes ------------
print("loading gt meshes")
gt_meshes = [my_load_mesh(s) for s in tqdm(range(model_idx, model_idx+1))]
timelapse.add_mesh_batch(category='gt_mesh',
                         vertices_list=[gt_meshes[0].vertices.cpu()],
                         faces_list=[gt_meshes[0].faces.cpu()])
occ_pts = torch.from_numpy(
    torch.argwhere(
        torch.from_numpy(occ[model_idx]).reshape([occ_res]*3) == 1.0).cpu().numpy())
occ_pts = occ_pts/occ_res - 0.5
timelapse.add_pointcloud_batch(category='gt_occ', pointcloud_list=[occ_pts])
# np.save(os.path.join(results_dir, 'gt_occ.npy'), occ[model_idx]); exit(0)

# ------------ embeddings ------------
occ_embeddings = torch.nn.Embedding(num_shapes, embed_dim).to(device)
torch.nn.init.normal_(occ_embeddings.weight.data, 0.0, 1 / math.sqrt(embed_dim))

# ------------ network ------------
occ_model = SmallMLPs(feature_dims=embed_dim,
                      internal_dims=128,
                      hidden=5,
                      multires=2).to(device)
occ_model_params = [p for _, p in occ_model.named_parameters()]

# ------------ optimizer ------------
optimizer = torch.optim.Adam([{"params": occ_model_params, "lr": lr},
                              {"params": occ_embeddings.parameters(), "lr": lr}])

# ------------ loss ------------
mse = torch.nn.MSELoss()
def loss_f(pred_values, gt_values):
    recon_loss = mse(pred_values, gt_values)
    return recon_loss

# ------------ batch loading ------------
def load_batch(batch_idx, batch_size, start=None, end=None):
    if start is None:
        start = batch_idx*batch_size
    if end is None:
        end = start + batch_size
    return occ_embeddings(torch.arange(start, end).to(device)),\
        torch.from_numpy(normalized_points[model_idx:model_idx+1]).to(device, torch.float32),\
        torch.from_numpy(values[model_idx:model_idx+1]).to(device, torch.float32),\
        [gt_meshes[i] for i in range(start, end)]

# ------------ reconstruction ------------
def export_mesh_normalized(mesh_name, vertices, faces):
    vertices = kaolin.ops.pointcloud.center_points(
        vertices.unsqueeze(0), normalize=True).squeeze(0)
    vertices = vertices.cpu().numpy()
    faces = faces.cpu().numpy()
    pred_mesh = trimesh.Trimesh(vertices, faces)
    pred_mesh.export(os.path.join(results_dir, f'{mesh_name}.obj'))
    return vertices, faces
def recon_occ_mesh(occ_model, occ_name, mesh_name):
    query_points = reconstruct.make_query_points(pt_sample_res, limits=[(-0.5, 0.5)]*3)
    query_points = torch.from_numpy(query_points).to(device, torch.float32).unsqueeze(0)
    occ_embed_feat, batch_points, batch_values, gt_meshes = load_batch(
        0, 0, start=0, end=1)
    with torch.no_grad():
        pred_occ = occ_model.forward(
            query_points,
            occ_embed_feat.unsqueeze(1).expand(-1, query_points.shape[1], -1)).detach()
    occ_grid = torch.reshape(
        pred_occ,
        (1, pt_sample_res, pt_sample_res, pt_sample_res))
    occ_grid = torch.permute(occ_grid, (0, 2, 1, 3))
    occ_path = os.path.join(results_dir, f'{occ_name}.npy')
    np.save(occ_path, occ_grid.cpu().numpy())
    pred_vertices, pred_faces =\
        kaolin.ops.conversions.voxelgrids_to_trianglemeshes(occ_grid)
    V, F = export_mesh_normalized(mesh_name, pred_vertices[0], pred_faces[0])

# ------------ flexicubes ------------
def run_flexi(sdf, gt_mesh, pred_occ):
    # NOTE: this chunk is crucial to keep training upon flexi injection!
    sdf_bxnxnxn = sdf.reshape((sdf.shape[0], fc_res+1, fc_res+1, fc_res+1))
    sdf_less_boundary = sdf_bxnxnxn[:, 1:-1, 1:-1, 1:-1].reshape(sdf.shape[0], -1)
    pos_shape = torch.sum((sdf_less_boundary > 0).int(), dim=-1)
    neg_shape = torch.sum((sdf_less_boundary < 0).int(), dim=-1)
    zero_surface = torch.bitwise_or(pos_shape == 0, neg_shape == 0)
    if torch.sum(zero_surface).item() > 0:
        update_sdf = torch.zeros_like(sdf[0:1])
        max_sdf = sdf.max()
        min_sdf = sdf.min()
        update_sdf[:, center_indices] += (1.0 - min_sdf)  # greater than zero
        update_sdf[:, boundary_indices] += (-1 - max_sdf)  # smaller than zero
        new_sdf = torch.zeros_like(sdf)
        for i_batch in range(zero_surface.shape[0]):
            if zero_surface[i_batch]:
                new_sdf[i_batch:i_batch + 1] += update_sdf
        update_mask = (new_sdf == 0).float()
        sdf = sdf * update_mask + new_sdf * (1 - update_mask)

    vertices, faces, L_dev = fc(
        x_nx3, sdf[0], cube_fx8, fc_res, training=True)
    flexicubes_mesh = util.Mesh(vertices, faces)

    mv, mvp = render.get_random_camera_batch(
        8, iter_res=train_res, device=device, use_kaolin=False)
    target = render.render_mesh_paper(gt_mesh, mv, mvp, train_res)

    # print(torch.mean(gt_mesh.vertices, dim=0))
    # exit(0)
    gt_mean = torch.mean(gt_mesh.vertices, dim=0)
    vertices = kaolin.ops.pointcloud.center_points(
        flexicubes_mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)
    flexi_mean = torch.mean(vertices, dim=0)

    try: 
        buffers = render.render_mesh_paper(flexicubes_mesh, mv, mvp, train_res)
    except Exception as e:
        import logging, traceback
        logging.error(traceback.format_exc())
        print(torch.min(sdf))
        print(torch.max(sdf))
        np.save(os.path.join(results_dir, 'badsdfout.npy'),
                sdf.detach().cpu().numpy())
        np.save(os.path.join(results_dir, 'badoccout.npy'),
                pred_occ.detach().cpu().numpy())
        exit(0)

    mask_loss = (buffers['mask'] - target['mask']).abs().mean()
    depth_loss = (((((buffers['depth'] - (target['depth']))*
                     target['mask'])**2).sum(-1)+1e-8)).sqrt().mean() * 10
    return mask_loss+depth_loss, vertices, faces, gt_mean, flexi_mean

# ------------ training ------------
if args.train:
    for it in range(iterations):
        optimizer.zero_grad()
        occ_embed_feat, batch_points, batch_values, gt_meshes = load_batch(
            0, 0, start=0, end=1)
        # don't use sigmoid to the output
        pred_occ = occ_model.forward(
            batch_points,
            occ_embed_feat.unsqueeze(1).expand(-1, batch_points.shape[1], -1))
        occ_loss = loss_f(pred_occ, batch_values)

        pred_verts_occ = occ_model.forward(
            flexi_verts,
            occ_embed_feat.unsqueeze(1).expand(-1, flexi_verts.shape[1], -1))
        comp_sdf = ops.bin2sdf_torch_3(pred_verts_occ)

        mesh_loss = 0
        center_loss = 0
        for s in range(batch_size):
            one_mesh_loss, vertices, faces, gt_mean, flexi_mean = run_flexi(
                torch.flatten(comp_sdf[s]).unsqueeze(0), gt_meshes[s],
                pred_verts_occ[s])
            mesh_loss += one_mesh_loss
            center_loss += loss_f(gt_mean, flexi_mean)
        total_loss = occ_loss + mesh_loss + center_loss

        total_loss.backward()
        optimizer.step()
        # scheduler.step()

        if (it) % 100 == 0 or it == (iterations - 1): 
            with torch.no_grad():
                vertices, faces, L_dev = fc(
                    x_nx3, torch.flatten(comp_sdf[-1]), cube_fx8,
                    fc_res, training=False)
            print(
                'Iteration {} - loss: {:.5f}, '.format(it, total_loss)+
                '# of mesh vertices: {}, # of mesh faces: {}'.format(
                    vertices.shape[0], faces.shape[0]))
            timelapse.add_mesh_batch(
                iteration=it+1,
                category='pred_mesh',
                vertices_list=[vertices.cpu()],
                faces_list=[faces.cpu()])
            grid = comp_sdf[-1].reshape(fc_res+1, fc_res+1, fc_res+1)
            occ_pts = torch.from_numpy(torch.argwhere(grid <= 0.0).cpu().numpy())
            occ_pts = occ_pts/(fc_res+1) - 0.5
            if occ_pts.shape[0] != 0:
                timelapse.add_pointcloud_batch(
                    iteration=it+1,
                    category='pred_occ',
                    pointcloud_list=[occ_pts])
            
recon_occ_mesh(occ_model, 'outocc', 'outoccmesh')
np.save(os.path.join(results_dir, 'outsdf.npy'), comp_sdf[0].detach().cpu().numpy())
V, F = export_mesh_normalized('outfleximesh', vertices.detach(), faces.detach())
