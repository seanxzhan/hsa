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
from utils import ops

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str)
parser.add_argument('--train', action="store_true")
parser.add_argument('--test', action="store_true")
parser.add_argument('--test_idx', '--ti', type=int)
parser.add_argument('--of', action="store_true", default=False)
parser.add_argument('--of_idx', '--oi', type=int)
parser.add_argument('--mask', '--mk', action="store_true")
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
lr = 0.01
iterations = 3000
train_res = [512, 512]
fc_voxel_grid_res = 31
occ_res = fc_voxel_grid_res + 1
model_idx = 0

partnet_dir = '/datasets/PartNet'
partnet_index_path = '/sota/partnet_dataset/stats/all_valid_anno_info.txt'
train_new_ids_to_objs_path = f'data/{cat_name}_train_new_ids_to_objs_18_{ds_start}_{ds_end}.json'
with open(train_new_ids_to_objs_path, 'r') as f:
    train_new_ids_to_objs: Dict = json.load(f)
model_idx_to_anno_id = {}
for mi, ai in enumerate(train_new_ids_to_objs.keys()):
    model_idx_to_anno_id[mi] = ai

results_dir = 'results/flexi-occsdf'
timelapse_dir = os.path.join(results_dir, 'training_timelapse')
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

gt_mesh = my_load_mesh(model_idx)
# gt_mesh = my_load_mesh(model_idx, tri=True)
# gt_mesh.export(os.path.join(results_dir, 'gt_mesh.obj'))
anno_id = model_idx_to_anno_id[model_idx]
timelapse.add_mesh_batch(category='gt_mesh',
                         vertices_list=[gt_mesh.vertices.cpu()],
                         faces_list=[gt_mesh.faces.cpu()])
np.save(os.path.join(results_dir, f'{anno_id}_gt_sdf.npy'), values[model_idx])
gt_sdf = torch.from_numpy(values[model_idx]).to(device)
np.save(os.path.join(results_dir, f'{anno_id}_gt_occ.npy'), occ[model_idx])
gt_occ = torch.from_numpy(occ[model_idx]).to(device)

fc = FlexiCubes(device)
x_nx3, cube_fx8 = fc.construct_voxel_grid(fc_voxel_grid_res)
# NOTE: 2* is necessary! Dunno why
batch_points = x_nx3.clone().to(device)
x_nx3 = 2*x_nx3
x_nx3 = x_nx3.clone().detach().requires_grad_(True)

# NOTE: testing network output
from occ_networks.xform_decoder_nasa_obbgnn_ae_58 import SDFDecoder
from utils import misc, reconstruct
occ_ckpt_path = 'logs/local_66-bs-25/64/ckpt/model_3000.pt'
occ_checkpoint = torch.load(occ_ckpt_path)
num_parts = 4
num_union_nodes = adj.shape[1]
occ_model = SDFDecoder(num_parts=4,
                       feature_dims=32,
                       internal_dims=128,
                       hidden=4,
                       multires=2).to(device)
occ_model.load_state_dict(occ_checkpoint['model_state_dict'])
occ_embeddings = torch.nn.Embedding(508, 128).to(device)
occ_embeddings.load_state_dict(occ_checkpoint['embeddings_state_dict'])

def run_occ(bs, occ_embed_feat, batch_points, batch_node_feat, batch_adj, batch_part_nodes,
            points_mask=None, parts_mask=None, occ_mask=None):
    batch_vec = torch.arange(start=0, end=bs).to(device)
    batch_vec = torch.repeat_interleave(batch_vec, num_union_nodes)
    batch_mask = torch.sum(batch_part_nodes, dim=1)
    with torch.no_grad():
        _, learned_xforms = occ_model.learn_geom_xform(batch_node_feat,
                                                        batch_adj,
                                                        batch_mask,
                                                        batch_vec)
    learned_xforms = learned_xforms[:, [0, 1, 2, 3], [0, 1, 2, 3], :]
    transformed_points = batch_points.unsqueeze(0).unsqueeze(0).expand(
        -1, num_parts, -1, -1) + learned_xforms.unsqueeze(2)
    
    with torch.no_grad():
        if not args.mask:
            pred_occ = occ_model(transformed_points, occ_embed_feat)
        else:
            pred_occ = occ_model(transformed_points.masked_fill(points_mask==1,torch.tensor(0)),
                                 occ_embed_feat.masked_fill(parts_mask==1, torch.tensor(0)))
            pred_occ = pred_occ.masked_fill(occ_mask==1,torch.tensor(float('-inf')))
        # loss_occ = loss_f(pred_occ, gt_occ[b*bs:(b+1)*bs])
    pred_occ, _ = torch.max(pred_occ, dim=-1, keepdim=True)
    return pred_occ

def load_batch(batch_idx, batch_size, start=None, end=None):
    if start is None:
        start = batch_idx*batch_size
    if end is None:
        end = start + batch_size
    return occ_embeddings(torch.arange(start, end).to(device)),\
        torch.from_numpy(node_features[start:end, :, :3]).to(device, torch.float32),\
        torch.from_numpy(adj[start:end]).to(device, torch.long),\
        torch.from_numpy(part_nodes[start:end]).to(device, torch.long),\

# comp_sdf = differentiable_sdf(gt_occ.reshape([occ_res, occ_res, occ_res]))
# comp_sdf = ops.bin2sdf(input=gt_occ.reshape([occ_res, occ_res, occ_res]).cpu().numpy())
# comp_sdf = torch.from_numpy(comp_sdf).to(device)
# comp_sdf = ops.bin2sdf_torch_1(input=gt_occ.reshape([occ_res, occ_res, occ_res]))
# comp_sdf = ops.bin2sdf_torch_2(input=gt_occ.reshape([occ_res, occ_res, occ_res]))
# comp_sdf = ops.bin2sdf_torch_3(input=gt_occ.reshape([occ_res, occ_res, occ_res]))
# comp_sdf = ops.bin2sdf_torch_4(input=gt_occ.reshape([occ_res, occ_res, occ_res]))

occ_embed_feat, batch_node_feat, batch_adj, batch_part_nodes = load_batch(0, 0, model_idx, model_idx+1)
# pred_occ = run_occ(1, occ_embed_feat, batch_points, batch_node_feat, batch_adj, batch_part_nodes).squeeze(0)

part = [1, 2, 3]

parts_mask = torch.zeros((1, num_parts)).to(device, torch.float32)
parts_mask[:, part] = 1
parts_mask = torch.repeat_interleave(parts_mask, 32, dim=-1)

points_mask = torch.zeros((1, num_parts, 1, 1)).to(device, torch.float32)
points_mask[:, part] = 1

occ_mask = torch.zeros((1, 1, num_parts)).to(device, torch.float32)
occ_mask[:, :, part] = 1

pred_occ = run_occ(1, occ_embed_feat, batch_points, batch_node_feat, batch_adj, batch_part_nodes,
                   points_mask=points_mask, parts_mask=parts_mask, occ_mask=occ_mask).squeeze(0)

plot_vals = False
if plot_vals:
    import matplotlib.pyplot as plt
    plt.hist(pred_occ.cpu().numpy().flatten(), bins=50, color='blue')
    plt.title("Histogram of Sampled SDF Values")
    plt.xlabel("SDF Value")
    plt.ylabel("Frequency")
    plt.xlim(-1, 1)
    misc.save_fig(plt, '', results_dir+'/partitioned_hist.png')
    plt.close()

# # pred_occ = (pred_occ > 0.5).float()
np.save(os.path.join(results_dir, f'{anno_id}_pred_occ.npy'), pred_occ.reshape([occ_res, occ_res, occ_res]).cpu().numpy())
# import time
# st = time.time()
comp_sdf = ops.bin2sdf_torch_3(input=pred_occ.reshape([occ_res, occ_res, occ_res]))
# print(time.time()-st)
# comp_sdf = comp_sdf.to(device)


np.save(os.path.join(results_dir, f'{anno_id}_comp_sdf.npy'), comp_sdf.cpu().numpy())
# exit(0)

# model = SDFDecoder(input_dims=3,
#                    num_parts=1,
#                    feature_dims=0,
#                    internal_dims=128,
#                    hidden=8,
#                    multires=2).to(device)
# params = [p for _, p in model.named_parameters()]
# model.pre_train_sphere(2000)

# sdf = torch.rand_like(x_nx3[:,0]) - 0.1 # randomly init SDF
# print(sdf.shape)
# sdf    = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
# deform = torch.nn.Parameter(torch.zeros_like(x_nx3), requires_grad=True)
# weight = torch.zeros((cube_fx8.shape[0], 21), dtype=torch.float, device='cuda') 
# weight    = torch.nn.Parameter(weight.clone().detach(), requires_grad=True)

# optimizer = torch.optim.Adam([sdf], lr=lr)
# # optimizer = torch.optim.Adam([sdf, deform], lr=lr)
# # optimizer = torch.optim.Adam([sdf, deform, weight], lr=lr)
# # optimizer = torch.optim.Adam(params=params, lr=lr)
# scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer, lr_lambda=lambda x: lr_schedule(x)) 

mse = torch.nn.MSELoss()
def loss_f(pred_values, gt_values):
    recon_loss = mse(pred_values, gt_values)
    return recon_loss

# -------- NOTE: testing flexicube's ability to convert sdf/psuedo sdf to mesh
grid_verts = x_nx3
vertices, faces, L_dev = fc(
    grid_verts, comp_sdf.flatten(), cube_fx8, fc_voxel_grid_res, training=False)
flexicubes_mesh = util.Mesh(vertices, faces)
np.save(os.path.join(results_dir, f'{anno_id}_flexi_sdf.npy'), comp_sdf.detach().cpu().numpy())
mesh_np = trimesh.Trimesh(vertices = vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), process=False)
mesh_np.export(os.path.join(results_dir, f'{anno_id}_flexi_mesh.obj'))

exit(0)

for it in range(iterations):
    optimizer.zero_grad()

    # model_out = model.get_sdf_deform(x_nx3)
    # NOTE: using tanh gives better results, training might be worse with smaller lr
    # sdf, deform = torch.tanh(model_out[:, :1]), model_out[:, 1:]
    # sdf, deform = model_out[:, :1], model_out[:, 1:]

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
    # grid_verts = x_nx3 + (2-1e-8) / (fc_voxel_grid_res * 2) * torch.tanh(deform)
    grid_verts = x_nx3
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
    
np.save(os.path.join(results_dir, f'{anno_id}_flexi_sdf.npy'), sdf.detach().cpu().numpy())
mesh_np = trimesh.Trimesh(vertices = vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), process=False)
mesh_np.export(os.path.join(results_dir, f'{anno_id}_flexi_mesh.obj'))