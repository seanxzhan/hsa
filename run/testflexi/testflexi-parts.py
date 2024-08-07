import os
import json
import h5py
import torch
import kaolin
import trimesh
import argparse
import numpy as np
from typing import Dict
from flexi.flexicubes import FlexiCubes
from flexi import render, util
from occ_networks.flexitest_decoder import SDFDecoder, get_embedder

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str)
parser.add_argument('--train', action="store_true")
parser.add_argument('--test', action="store_true")
parser.add_argument('--test_idx', '--ti', type=int)
parser.add_argument('--of', action="store_true", default=False)
parser.add_argument('--of_idx', '--oi', type=int)
args = parser.parse_args()

name_to_cat = {
    'Chair': '03001627',
    'Lamp': '03636649'
}
cat_name = 'Chair'
pt_sample_res = 64
ds_start, ds_end = 0, 10
OVERFIT = args.of
overfit_idx = args.of_idx
device = 'cuda'
lr = 0.001
iterations = 3000
train_res = [512, 512]
fc_voxel_grid_res = 31
model_idx = 0
part = 1

partnet_dir = '/datasets/PartNet'
partnet_index_path = '/sota/partnet_dataset/stats/all_valid_anno_info.txt'
train_new_ids_to_objs_path = f'data/{cat_name}_train_new_ids_to_objs_18_{ds_start}_{ds_end}.json'
with open(train_new_ids_to_objs_path, 'r') as f:
    train_new_ids_to_objs: Dict = json.load(f)
model_idx_to_anno_id = {}
for mi, ai in enumerate(train_new_ids_to_objs.keys()):
    model_idx_to_anno_id[mi] = ai

anno_id = model_idx_to_anno_id[model_idx]

# results_dir = 'results/flexi-parts'
results_dir = os.path.join('results', 'flexi-parts', anno_id)
timelapse_dir = os.path.join(results_dir, f'training_timelapse_{part}')
print("timelapse dir: ", timelapse_dir)
timelapse = kaolin.visualize.Timelapse(timelapse_dir)

train_data_path = f'data/{cat_name}_train_{pt_sample_res}_18_{ds_start}_{ds_end}.hdf5'
train_data = h5py.File(train_data_path, 'r')
if OVERFIT:
    part_num_indices = train_data['part_num_indices'][overfit_idx:overfit_idx+1]
    all_indices = train_data['all_indices'][overfit_idx:overfit_idx+1]
    normalized_points = train_data['normalized_points'][overfit_idx:overfit_idx+1]
    values = train_data['values'][overfit_idx:overfit_idx+1]
    occ = train_data['occ'][overfit_idx:overfit_idx+1]
    part_nodes = train_data['part_nodes'][overfit_idx:overfit_idx+1]
    xforms = train_data['xforms'][overfit_idx:overfit_idx+1]
    extents = train_data['extents'][overfit_idx:overfit_idx+1]
    node_features = train_data['node_features'][overfit_idx:overfit_idx+1]
    adj = train_data['adj'][overfit_idx:overfit_idx+1]
    part_nodes = train_data['part_nodes'][overfit_idx:overfit_idx+1]
    relations = train_data['relations'][overfit_idx:overfit_idx+1]
else:
    part_num_indices = train_data['part_num_indices']
    all_indices = train_data['all_indices']
    normalized_points = train_data['normalized_points']
    values = train_data['values']
    occ = train_data['occ']
    part_nodes = train_data['part_nodes']
    xforms = train_data['xforms']
    extents = train_data['extents']
    node_features = train_data['node_features']
    adj = train_data['adj']
    part_nodes = train_data['part_nodes']
    relations = train_data['relations']
    part_verts = [train_data['part_verts_0'],
                  train_data['part_verts_1'],
                  train_data['part_verts_2'],
                  train_data['part_verts_3']]
    part_faces = [train_data['part_faces_0'],
                  train_data['part_faces_1'],
                  train_data['part_faces_2'],
                  train_data['part_faces_3']]

def lr_schedule(iter):
    return max(0.0, 10**(-(iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.    

def my_load_mesh(model_idx, tri=False):
    anno_id = model_idx_to_anno_id[model_idx]
    obj_dir = os.path.join(partnet_dir, anno_id, 'vox_models')
    assert os.path.exists(obj_dir)
    gt_mesh_path = os.path.join(obj_dir, f'{anno_id}.obj')
    gt_mesh = trimesh.load_mesh(gt_mesh_path)
    gt_vertices_aligned = torch.from_numpy(
        gt_mesh.vertices).to(device).to(torch.float32)
    gt_vertices_aligned = kaolin.ops.pointcloud.center_points(
        gt_vertices_aligned.unsqueeze(0), normalize=True).squeeze(0)
    gt_vertices_aligned = gt_vertices_aligned.cpu().numpy()
    if not tri:
        return util.load_mesh_vf(gt_vertices_aligned, gt_mesh.faces, device)
    else:
        return trimesh.Trimesh(gt_vertices_aligned, gt_mesh.faces)
    
def my_load_mesh_vf(model_idx, part):
    verts = np.array(part_verts[part][model_idx]).reshape(-1, 3)
    faces = np.array(part_faces[part][model_idx]).reshape(-1, 3)
    return util.load_mesh_vf(verts, faces, device)

# gt_mesh = my_load_mesh(model_idx)
gt_mesh = my_load_mesh_vf(model_idx, part)
# gt_mesh = my_load_mesh(model_idx, tri=True)
# gt_mesh.export(os.path.join(results_dir, 'gt_mesh.obj'))
timelapse.add_mesh_batch(category='gt_mesh',
                         vertices_list=[gt_mesh.vertices.cpu()],
                         faces_list=[gt_mesh.faces.cpu()])
np.save(os.path.join(results_dir, f'{anno_id}_gt_sdf_{part}.npy'), values[model_idx])
gt_sdf = torch.from_numpy(values[model_idx]).to(device)

fc = FlexiCubes(device)
x_nx3, cube_fx8 = fc.construct_voxel_grid(fc_voxel_grid_res)
# NOTE: 2* is necessary! Dunno why
x_nx3 = 2*x_nx3

x_nx3 = x_nx3.clone().detach().requires_grad_(True)

model = SDFDecoder(input_dims=3,
                   num_parts=1,
                   feature_dims=0,
                   internal_dims=128,
                   hidden=8,
                   multires=2).to(device)
params = [p for _, p in model.named_parameters()]
# model.pre_train_sphere(2000)

sdf = torch.rand_like(x_nx3[:,0]) - 0.1 # randomly init SDF
sdf    = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
deform = torch.nn.Parameter(torch.zeros_like(x_nx3), requires_grad=True)
# weight = torch.zeros((cube_fx8.shape[0], 21), dtype=torch.float, device='cuda') 
# weight    = torch.nn.Parameter(weight.clone().detach(), requires_grad=True)

optimizer = torch.optim.Adam([sdf, deform], lr=lr)
# optimizer = torch.optim.Adam([sdf, deform, weight], lr=lr)
# optimizer = torch.optim.Adam(params=params, lr=lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda x: lr_schedule(x)) 

mse = torch.nn.MSELoss()
def loss_f(pred_values, gt_values):
    recon_loss = mse(pred_values, gt_values)
    return recon_loss

def get_center_boundary_index(grid_res, device):
    v = torch.zeros((grid_res + 1, grid_res + 1, grid_res + 1), dtype=torch.bool, device=device)
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
center_indices, boundary_indices = get_center_boundary_index(fc_voxel_grid_res, device)


for it in range(iterations):
    optimizer.zero_grad()

    model_out = model.get_sdf_deform(x_nx3)
    # NOTE: using tanh gives better results, training might be worse with smaller lr
    # sdf, deform = torch.tanh(model_out[:, :1]), model_out[:, 1:]
    sdf, deform = model_out[:, :1], model_out[:, 1:]

    sdf = sdf.unsqueeze(0)
    # print(sdf.shape)
    # exit(0)
    # deformation = 1.0 / (fc_voxel_grid_res * 4) * torch.tanh(deform)
    
    sdf_bxnxnxn = sdf.reshape((sdf.shape[0], fc_voxel_grid_res + 1, fc_voxel_grid_res + 1, fc_voxel_grid_res + 1))
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
        # # Regulraization here is used to push the sdf to be a different sign (make it not fully positive or fully negative)
        # sdf_reg_loss = torch.abs(sdf).mean(dim=-1).mean(dim=-1)
        # sdf_reg_loss = sdf_reg_loss * zero_surface.float()
        sdf = sdf * update_mask + new_sdf * (1 - update_mask)

    sdf = sdf.squeeze(0)

    # NOTE: don't use ground truth SDF supervision
    # total_loss = loss_f(sdf, gt_sdf)
    # if it <= 500:
    #     total_loss = loss_f(sdf[::100], gt_sdf[::100])
    #     # NOTE: initializing SDF this way is necessary to get training started
    #     if (it) % 100 == 0 or it == (iterations - 1): 
    #         print('Iteration {} - loss: {:.5f}'.format(it, total_loss))
    #     total_loss.backward()
    #     optimizer.step()
    #     scheduler.step()
    #     continue

    # NOTE: eikonal loss does make SDF smoother, but final result isn't as great
    # gradient = torch.autograd.grad(outputs=sdf, inputs=x_nx3,
    #                                grad_outputs=torch.ones_like(sdf),
    #                                create_graph=True, retain_graph=True)[0]
    # grad_norm = torch.norm(gradient, dim=-1)
    # eikonal_loss = ((grad_norm - 1.0) ** 2).mean()

    mv, mvp = render.get_random_camera_batch(
        8, iter_res=train_res, device=device, use_kaolin=False)
    target = render.render_mesh_paper(gt_mesh, mv, mvp, train_res)
    grid_verts = x_nx3 + (2-1e-8) / (fc_voxel_grid_res * 2) * torch.tanh(deform)
    vertices, faces, L_dev = fc(
        grid_verts, sdf, cube_fx8, fc_voxel_grid_res, training=True)
    # vertices, faces, L_dev = fc(
    #     grid_verts, sdf, cube_fx8, fc_voxel_grid_res,
    #     beta_fx12=weight[:,:12], alpha_fx8=weight[:,12:20],
    #     gamma_f=weight[:,20], training=True)
    flexicubes_mesh = util.Mesh(vertices, faces)
    try: 
        buffers = render.render_mesh_paper(flexicubes_mesh, mv, mvp, train_res)
    except Exception as e:
        import logging, traceback
        logging.error(traceback.format_exc())
        print(torch.min(sdf))
        print(torch.max(sdf))
        exit(0)
    mask_loss = (buffers['mask'] - target['mask']).abs().mean()
    depth_loss = (((((buffers['depth'] - (target['depth']))* target['mask'])**2).sum(-1)+1e-8)).sqrt().mean() * 10

    total_loss = mask_loss + depth_loss
    # total_loss = mask_loss + depth_loss + eikonal_loss

    total_loss.backward()
    optimizer.step()
    scheduler.step()

    if (it) % 100 == 0 or it == (iterations - 1): 
        with torch.no_grad():
            vertices, faces, L_dev = fc(
                grid_verts, sdf, cube_fx8, fc_voxel_grid_res,
                training=False)
            # vertices, faces, L_dev = fc(
            #     grid_verts, sdf, cube_fx8, fc_voxel_grid_res,
            #     beta_fx12=weight[:,:12], alpha_fx8=weight[:,12:20],
            #     gamma_f=weight[:,20], training=False)
        print ('Iteration {} - loss: {:.5f}, # of mesh vertices: {}, # of mesh faces: {}'.format(
            it, total_loss, vertices.shape[0], faces.shape[0]))
        # save reconstructed mesh
        timelapse.add_mesh_batch(
            iteration=it+1,
            category='extracted_mesh',
            vertices_list=[vertices.cpu()],
            faces_list=[faces.cpu()]
        )
    
np.save(os.path.join(results_dir, f'{anno_id}_flexi_sdf_{part}.npy'), sdf.detach().cpu().numpy())
mesh_np = trimesh.Trimesh(vertices = vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), process=False)
mesh_np.export(os.path.join(results_dir, f'{anno_id}_flexi_mesh_{part}.obj'))