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
from occ_networks.flexi_decoder_27 import SDFDecoder as FlexiDecoder
from occ_networks.xform_decoder_nasa_obbgnn_ae_58 import SDFDecoder, get_embedder
from utils import misc, reconstruct

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str)
parser.add_argument('--train', action="store_true")
parser.add_argument('--test', action="store_true")
parser.add_argument('--it', type=int)
parser.add_argument('--test_idx', '--ti', type=int)
parser.add_argument('--of', action="store_true", default=False)
parser.add_argument('--of_idx', '--oi', type=int)
parser.add_argument('--mask', '--mk', action="store_true")
parser.add_argument('--inversion', '--inv', action="store_true")
args = parser.parse_args()

name_to_cat = {
    'Chair': '03001627',
    'Lamp': '03636649'
}
cat_name = 'Chair'
pt_sample_res = 64
ds_start, ds_end = 0, 508
OVERFIT = args.of
overfit_idx = args.of_idx
device = 'cuda'
lr = 0.01
# lr = 0.005
iterations = 10000; iterations += 1
train_res = [512, 512]
fc_voxel_grid_res = 31
# model_idx = 2
expt_id = 33

# ------------ data dirs ------------
partnet_dir = '/datasets/PartNet'
partnet_index_path = '/sota/partnet_dataset/stats/all_valid_anno_info.txt'
train_new_ids_to_objs_path = f'data/{cat_name}_train_new_ids_to_objs_12_{ds_start}_{ds_end}.json'
with open(train_new_ids_to_objs_path, 'r') as f:
    train_new_ids_to_objs: Dict = json.load(f)
model_idx_to_anno_id = {}
for mi, ai in enumerate(train_new_ids_to_objs.keys()):
    model_idx_to_anno_id[mi] = ai

# ------------ logging ------------
results_dir = f'results/flexi_{expt_id}'
timelapse_dir = os.path.join(results_dir, 'training_timelapse')
print("timelapse dir: ", timelapse_dir)
timelapse = kaolin.visualize.Timelapse(timelapse_dir)
logs_path = os.path.join('logs', 'flexi', f'flexi_{expt_id}')
ckpt_dir = os.path.join(logs_path, 'ckpt'); misc.check_dir(ckpt_dir)
writer = SummaryWriter(os.path.join(logs_path, 'summary'))

# ------------ data loading ------------
train_data_path = f'data/{cat_name}_train_{pt_sample_res}_12_{ds_start}_{ds_end}.hdf5'
train_data = h5py.File(train_data_path, 'r')
if OVERFIT:
    part_num_indices = train_data['part_num_indices'][overfit_idx:overfit_idx+1]
    all_indices = train_data['all_indices'][overfit_idx:overfit_idx+1]
    normalized_points = train_data['normalized_points'][overfit_idx:overfit_idx+1]
    values = train_data['values'][overfit_idx:overfit_idx+1]
    # occ = train_data['occ'][overfit_idx:overfit_idx+1]
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
    # occ = train_data['occ']
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

num_parts = 4
num_shapes = 10
bs = 10
nb = 1
num_union_nodes = adj.shape[1]

# ------------ init flexicubes ------------
fc = FlexiCubes(device)
x_nx3, cube_fx8 = fc.construct_voxel_grid(fc_voxel_grid_res)
# NOTE: for occ inference, range needs to be -.5 to .5
batch_points = x_nx3.clone().to(device)
# NOTE: 2* is necessary! Dunno why
x_nx3 = 2*x_nx3
x_nx3 = x_nx3.clone().detach().requires_grad_(True)

# ------------ init gt meshes ------------
# gt_occ = torch.from_numpy(occ[0:num_shapes]).to(device)
gt_meshes = [my_load_mesh(s) for s in range(0, num_shapes)]

# ------------ occ model and embeddings ------------
occ_ckpt_path = 'logs/local_66-bs-25/64/ckpt/model_3000.pt'
occ_checkpoint = torch.load(occ_ckpt_path)
occ_model = SDFDecoder(num_parts=num_parts,
                       feature_dims=32,
                       internal_dims=128,
                       hidden=4,
                       multires=2).to(device)
occ_model.load_state_dict(occ_checkpoint['model_state_dict'])
occ_embeddings = torch.nn.Embedding(ds_end-ds_start, 128).to(device)
occ_embeddings.load_state_dict(occ_checkpoint['embeddings_state_dict'])

# ------------ embeddings ------------
# embed_dim = 128
# embeddings = torch.nn.Embedding(num_shapes, embed_dim).to(device)
# torch.nn.init.normal_(embeddings.weight.data, 0.0, 1 / math.sqrt(embed_dim))

# ------------ flexi network ------------
model = FlexiDecoder(input_dims=3,
                     num_parts=num_parts,
                     feature_dims=128,
                     internal_dims=128,
                     hidden=5,
                     multires=2).to(device)
params = [p for _, p in model.named_parameters()]
# model.pre_train_sphere(1000) if args.train else None

# ------------ optimizer ------------
# optimizer = torch.optim.Adam([{"params": params, "lr": lr},
#                               {"params": embeddings.parameters(), "lr": lr}])
optimizer = torch.optim.Adam(params, lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda x: lr_schedule(x))

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
        torch.from_numpy(node_features[start:end, :, :3]).to(device, torch.float32),\
        torch.from_numpy(adj[start:end]).to(device, torch.long),\
        torch.from_numpy(part_nodes[start:end]).to(device, torch.long),\

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

def run_flexi(gt_mesh_idx, s, sdf_deform, pred_occ, gt_meshes, test=False):
    # NOTE: using tanh gives slightly better SDF results, still scattered, but -1~1
    # NOTE: not using tanh gives better final results, range very large
    # delta_sdf, deform = torch.tanh(model_out[s, :, :1]), model_out[s, :, 1:]
    delta_sdf, deform = sdf_deform[s, :, :1], sdf_deform[s, :, 1:]
    sdf = -2*pred_occ[s] + 1 + delta_sdf
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

    # if not test:
    #     gradient = torch.autograd.grad(outputs=sdf, inputs=x_nx3,
    #                                     grad_outputs=torch.ones_like(sdf),
    #                                     create_graph=True, retain_graph=True)[0]
    #     grad_norm = torch.norm(gradient, dim=-1)
    #     eikonal_loss = ((grad_norm - 1.0) ** 2).mean()
    # else:
    #     eikonal_loss = 0

    mv, mvp = render.get_random_camera_batch(
        8, iter_res=train_res, device=device, use_kaolin=False)
    target = render.render_mesh_paper(gt_meshes[gt_mesh_idx], mv, mvp, train_res)
    grid_verts = x_nx3 + (2-1e-8) / (fc_voxel_grid_res * 2) * torch.tanh(deform)
    vertices, faces, L_dev = fc(
        grid_verts, sdf, cube_fx8, fc_voxel_grid_res, training=True)
    flexicubes_mesh = util.Mesh(vertices, faces)
    try: 
        buffers = render.render_mesh_paper(flexicubes_mesh, mv, mvp, train_res)
    except Exception as e:
        import logging, traceback
        logging.error(traceback.format_exc())
        print(torch.min(sdf))
        print(torch.max(sdf))
        exit(0)
    
    if not test:
        mask_loss = (buffers['mask'] - target['mask']).abs().mean()
        depth_loss = (((((buffers['depth'] - (target['depth']))* target['mask'])**2).sum(-1)+1e-8)).sqrt().mean() * 10
    else:
        mask_loss = 0
        depth_loss = 0

    # return eikonal_loss+mask_loss+depth_loss, grid_verts, sdf, vertices, faces
    return mask_loss+depth_loss, grid_verts, sdf, vertices, faces

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

if args.train:
    for it in range(iterations):
        itr_loss = 0
        # itr_occ_loss = 0
        # itr_mesh_loss = 0
        for b in range(nb):
            optimizer.zero_grad()

            occ_embed_feat, batch_node_feat, batch_adj, batch_part_nodes = load_batch(b, bs)
            pred_occ = run_occ(bs, occ_embed_feat, batch_points,
                               batch_node_feat, batch_adj, batch_part_nodes)

            sdf_deform = model.get_sdf_deform(x_nx3, occ_embed_feat)
            
            all_loss = 0
            for s in range(10):
                one_mesh_loss, grid_verts, sdf, vertices, faces =\
                    run_flexi(b*bs+s, s, sdf_deform, pred_occ, gt_meshes)
                all_loss += one_mesh_loss

            # mesh_loss = all_loss / bs
            total_loss = all_loss / bs

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            itr_loss += total_loss
            # itr_occ_loss += loss_occ
            # itr_mesh_loss += mesh_loss

        itr_loss /= nb
        # itr_occ_loss /= nb
        # itr_mesh_loss /= nb

        writer.add_scalar('iteration loss', itr_loss, it)
        # writer.add_scalar('occ loss', itr_occ_loss, it)
        # writer.add_scalar('mesh loss', itr_mesh_loss, it)

        if (it) % 20 == 0 or it == (iterations - 1): 
            with torch.no_grad():
                vertices, faces, L_dev = fc(
                    grid_verts, sdf, cube_fx8, fc_voxel_grid_res,
                    training=False)
            print ('Iteration {} - loss: {:.5f}, # of mesh vertices: {}, # of mesh faces: {}'.format(
                it, total_loss, vertices.shape[0], faces.shape[0]))
            # save reconstructed mesh
            timelapse.add_mesh_batch(
                iteration=it+1,
                category='extracted_mesh',
                vertices_list=[vertices.cpu()],
                faces_list=[faces.cpu()]
            )
        if (it) % 200 == 0 or it == (iterations - 1): 
            torch.save({
                'epoch': it,
                'model_state_dict': model.state_dict(),
                # 'embeddings_state_dict': embeddings.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_loss': total_loss,
                }, os.path.join(ckpt_dir, f'model_{it}.pt'))
    writer.close()


if args.test:
    it = args.it
    model_idx = args.test_idx
    anno_id = model_idx_to_anno_id[model_idx]

    results_dir = os.path.join(results_dir, anno_id)
    misc.check_dir(results_dir)
    print("results dir: ", results_dir)

    checkpoint = torch.load(os.path.join(ckpt_dir, f'model_{it}.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    occ_embed_feat, batch_node_feat, batch_adj, batch_part_nodes =\
        load_batch(0, 0, model_idx, model_idx+1)
    
    query_points = reconstruct.make_query_points(pt_sample_res,
                                                 limits=[(-0.5, 0.5)]*3)
    query_points = torch.from_numpy(query_points).to(device, torch.float32)

    print("running inference")
    with torch.no_grad():
        pred_occ_grid = run_occ(1, occ_embed_feat, query_points, batch_node_feat, batch_adj, batch_part_nodes)
        pred_occ = run_occ(1, occ_embed_feat, batch_points, batch_node_feat, batch_adj, batch_part_nodes)
        sdf_deform = model.get_sdf_deform(x_nx3, occ_embed_feat)        
        _, _, sdf, vertices, faces =\
            run_flexi(model_idx, 0, sdf_deform, pred_occ, gt_meshes, test=True)

    from utils import visualize    
    gt_color = [31, 119, 180, 255]
    mesh_pred_path = os.path.join(results_dir, 'mesh_pred.png')
    mesh_flexi_path = os.path.join(results_dir, 'mesh_flexi.png')
    mesh_gt_path = os.path.join(results_dir, 'mesh_gt.png')
    lst_paths = [
        mesh_gt_path,
        mesh_pred_path,
        mesh_flexi_path]

    mag = 0.8; white_bg = False

    # ------------ gt ------------
    print("visualizing gt mesh")
    obj_dir = os.path.join(partnet_dir, anno_id, 'vox_models')
    assert os.path.exists(obj_dir)
    gt_mesh_path = os.path.join(obj_dir, f'{anno_id}.obj')
    gt_mesh = trimesh.load(gt_mesh_path, file_type='obj', force='mesh')
    gt_vertices_aligned = torch.from_numpy(
        gt_mesh.vertices).to(device).to(torch.float32)
    gt_vertices_aligned = kaolin.ops.pointcloud.center_points(
        gt_vertices_aligned.unsqueeze(0), normalize=True).squeeze(0)
    gt_vertices_aligned = gt_vertices_aligned.cpu().numpy()
    aligned_gt_mesh = trimesh.Trimesh(gt_vertices_aligned, gt_mesh.faces)
    aligned_gt_mesh.export(os.path.join(results_dir, 
                                        f'{anno_id}_mesh_gt_{expt_id}.obj'))
    aligned_gt_mesh.visual.vertex_colors = gt_color
    gt_mesh_vis = visualize.save_mesh_vis(aligned_gt_mesh, mesh_gt_path,
                                          mag=mag, white_bg=white_bg,
                                          save_img=False)
    
    # ------------ occ ------------
    print("visualizing pred mesh")
    occ_grid = torch.reshape(
        pred_occ_grid,
        (1, pt_sample_res, pt_sample_res, pt_sample_res))
    occ_grid = torch.permute(occ_grid, (0, 2, 1, 3))
    pred_vertices, pred_faces =\
        kaolin.ops.conversions.voxelgrids_to_trianglemeshes(occ_grid)
    pred_vertices = kaolin.ops.pointcloud.center_points(
        pred_vertices[0].unsqueeze(0), normalize=True).squeeze(0)
    pred_vertices = pred_vertices.cpu().numpy()
    pred_faces = pred_faces[0].cpu().numpy()
    pred_mesh = trimesh.Trimesh(pred_vertices, pred_faces)
    pred_mesh.export(os.path.join(results_dir, 
                                  f'{anno_id}_mesh_pred_{expt_id}.obj'))
    pred_mesh_vis = visualize.save_mesh_vis(pred_mesh, mesh_pred_path,
                                            mag=mag, white_bg=white_bg,
                                            save_img=False)

    # ------------ flexi ------------
    print("visualizing flexi mesh")
    flexi_faces = faces.detach().cpu().numpy()
    flexi_vertices_aligned = kaolin.ops.pointcloud.center_points(
        vertices.detach().unsqueeze(0), normalize=True).squeeze(0)
    flexi_vertices_aligned = flexi_vertices_aligned.cpu().numpy()
    flexi_mesh = trimesh.Trimesh(flexi_vertices_aligned, flexi_faces)
    flexi_mesh.export(os.path.join(results_dir,
                                   f'{anno_id}_flexi_mesh_{expt_id}.obj'))
    flexi_mesh_vis = visualize.save_mesh_vis(flexi_mesh, mesh_flexi_path,
                                             mag=mag, white_bg=white_bg,
                                             save_img=False)
    print("saving flexi sdf")
    np.save(os.path.join(results_dir, 
                         f'{anno_id}_flexi_sdf_{expt_id}.npy'),
            sdf.detach().cpu().numpy())
    
    print("saving images")
    visualize.stitch_imges(
        os.path.join(results_dir,f'{anno_id}_results_{expt_id}.png'),
        images=[gt_mesh_vis, pred_mesh_vis, flexi_mesh_vis],
        adj=100)


if args.mask:
    it = args.it
    model_idx = args.test_idx
    anno_id = model_idx_to_anno_id[model_idx]

    results_dir = os.path.join(results_dir, 'mask', anno_id)
    misc.check_dir(results_dir)
    print("results dir: ", results_dir)

    checkpoint = torch.load(os.path.join(ckpt_dir, f'model_{it}.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    occ_embed_feat, batch_node_feat, batch_adj, batch_part_nodes =\
        load_batch(0, 0, model_idx, model_idx+1)
    
    query_points = reconstruct.make_query_points(pt_sample_res,
                                                 limits=[(-0.5, 0.5)]*3)
    query_points = torch.from_numpy(query_points).to(device, torch.float32)

    gt_meshes = [my_load_mesh(model_idx)]

    # occ_embed_feat[:,3*32:4*32] = 0
    # print(occ_embed_feat.shape)
    # exit(0)

    part = 0

    parts_mask = torch.zeros((1, num_parts)).to(device, torch.float32)
    parts_mask[:, part] = 1
    parts_mask = torch.repeat_interleave(parts_mask, 32, dim=-1)

    points_mask = torch.zeros((1, num_parts, 1, 1)).to(device, torch.float32)
    points_mask[:, part] = 1

    occ_mask = torch.zeros((1, 1, num_parts)).to(device, torch.float32)
    occ_mask[:, :, part] = 1

    print("running inference")
    with torch.no_grad():
        pred_occ_grid = run_occ(1, occ_embed_feat, query_points, batch_node_feat, batch_adj, batch_part_nodes,
                                points_mask=points_mask, parts_mask=parts_mask, occ_mask=occ_mask)
        pred_occ = run_occ(1, occ_embed_feat, batch_points, batch_node_feat, batch_adj, batch_part_nodes,
                           points_mask=points_mask, parts_mask=parts_mask, occ_mask=occ_mask)
        sdf_deform = model.get_sdf_deform(x_nx3, occ_embed_feat)        
        _, _, sdf, vertices, faces =\
            run_flexi(0, 0, sdf_deform, pred_occ, gt_meshes, test=True)

    from utils import visualize    
    gt_color = [31, 119, 180, 255]
    mesh_pred_path = os.path.join(results_dir, 'mesh_pred.png')
    mesh_flexi_path = os.path.join(results_dir, 'mesh_flexi.png')
    mesh_gt_path = os.path.join(results_dir, 'mesh_gt.png')
    lst_paths = [
        mesh_gt_path,
        mesh_pred_path,
        mesh_flexi_path]

    mag = 0.8; white_bg = False

    # ------------ gt ------------
    print("visualizing gt mesh")
    obj_dir = os.path.join(partnet_dir, anno_id, 'vox_models')
    assert os.path.exists(obj_dir)
    gt_mesh_path = os.path.join(obj_dir, f'{anno_id}.obj')
    gt_mesh = trimesh.load(gt_mesh_path, file_type='obj', force='mesh')
    gt_vertices_aligned = torch.from_numpy(
        gt_mesh.vertices).to(device).to(torch.float32)
    gt_vertices_aligned = kaolin.ops.pointcloud.center_points(
        gt_vertices_aligned.unsqueeze(0), normalize=True).squeeze(0)
    gt_vertices_aligned = gt_vertices_aligned.cpu().numpy()
    aligned_gt_mesh = trimesh.Trimesh(gt_vertices_aligned, gt_mesh.faces)
    aligned_gt_mesh.export(os.path.join(results_dir, 
                                        f'{anno_id}_mesh_gt_{expt_id}.obj'))
    aligned_gt_mesh.visual.vertex_colors = gt_color
    gt_mesh_vis = visualize.save_mesh_vis(aligned_gt_mesh, mesh_gt_path,
                                          mag=mag, white_bg=white_bg,
                                          save_img=False)
    
    # ------------ occ ------------
    print("visualizing pred mesh")
    occ_grid = torch.reshape(
        pred_occ_grid,
        (1, pt_sample_res, pt_sample_res, pt_sample_res))
    occ_grid = torch.permute(occ_grid, (0, 2, 1, 3))
    pred_vertices, pred_faces =\
        kaolin.ops.conversions.voxelgrids_to_trianglemeshes(occ_grid)
    pred_vertices = kaolin.ops.pointcloud.center_points(
        pred_vertices[0].unsqueeze(0), normalize=True).squeeze(0)
    pred_vertices = pred_vertices.cpu().numpy()
    pred_faces = pred_faces[0].cpu().numpy()
    pred_mesh = trimesh.Trimesh(pred_vertices, pred_faces)
    pred_mesh.export(os.path.join(results_dir, 
                                  f'{anno_id}_mesh_pred_{expt_id}.obj'))
    pred_mesh_vis = visualize.save_mesh_vis(pred_mesh, mesh_pred_path,
                                            mag=mag, white_bg=white_bg,
                                            save_img=False)

    # ------------ flexi ------------
    print("visualizing flexi mesh")
    flexi_faces = faces.detach().cpu().numpy()
    flexi_vertices_aligned = kaolin.ops.pointcloud.center_points(
        vertices.detach().unsqueeze(0), normalize=True).squeeze(0)
    flexi_vertices_aligned = flexi_vertices_aligned.cpu().numpy()
    flexi_mesh = trimesh.Trimesh(flexi_vertices_aligned, flexi_faces)
    flexi_mesh.export(os.path.join(results_dir,
                                   f'{anno_id}_flexi_mesh_{expt_id}.obj'))
    flexi_mesh_vis = visualize.save_mesh_vis(flexi_mesh, mesh_flexi_path,
                                             mag=mag, white_bg=white_bg,
                                             save_img=False)
    print("saving flexi sdf")
    np.save(os.path.join(results_dir, 
                         f'{anno_id}_flexi_sdf_{expt_id}.npy'),
            sdf.detach().cpu().numpy())
    
    print("saving images")
    visualize.stitch_imges(
        os.path.join(results_dir,f'{anno_id}_results_{expt_id}.png'),
        images=[gt_mesh_vis, pred_mesh_vis, flexi_mesh_vis],
        adj=100)




# # ------------ inversion. ------------
# if args.inversion:
#     it = args.it
#     model_idx = args.test_idx
#     anno_id = model_idx_to_anno_id[model_idx]

#     results_dir = os.path.join(results_dir, 'inv', anno_id)
#     misc.check_dir(results_dir)
#     print("results dir: ", results_dir)
#     checkpoint = torch.load(os.path.join(ckpt_dir, f'model_{it}.pt'))
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()

#     # ------------ data ------------
#     query_points = reconstruct.make_query_points(pt_sample_res,
#                                                  limits=[(-1, 1)]*3)
#     query_points = torch.from_numpy(query_points).to(device, torch.float32)
#     query_points = query_points[None, None].expand(1, num_parts, -1, -1)
#     transformed_points = batch_points[None, None].expand(1, num_parts, -1, -1)

#     # ------------ init gt occ and mesh ------------
#     # gt_occ = torch.from_numpy(occ[model_idx:model_idx+1]).to(device)
#     gt_meshes = [my_load_mesh(model_idx)]

#     # ------------ embedding ------------
#     embedding_fp = os.path.join(results_dir, 'embedding.pth')
#     if os.path.exists(embedding_fp):
#         embeddings = torch.load(embedding_fp).to(device)
#     else:
#         embeddings = torch.nn.Embedding(1, 128).to(device)
#         torch.nn.init.normal_(embeddings.weight.data,
#                               0.0, 1 / math.sqrt(128))

#     # ------------ training ------------
#     optimizer = torch.optim.Adam([{"params": embeddings.parameters(), "lr": lr}])
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
#                                                   lr_lambda=lambda x: lr_schedule(x))
#     iterations = 1000

#     # ------------ optimize for embedding ------------
#     if not os.path.exists(embedding_fp):
#         print("optimizing for embedding")
#         for i in tqdm(range(iterations)):
#             optimizer.zero_grad()
#             embed_feat = embeddings(torch.tensor(0).to(device)).unsqueeze(0)
#             # with torch.no_grad():
#             pred_occ_grid = model.get_occ(transformed_points, embed_feat)
#             pred_occ = model.get_occ(transformed_points, embed_feat)
#             sdf_deform = model.get_sdf_deform(x_nx3, embed_feat)
#             one_mesh_loss, _, _, _, _ = run_flexi(
#                 0, 0, sdf_deform, pred_occ, gt_meshes)
#             # loss = loss_f(pred_occ_grid, gt_occ) + one_mesh_loss
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#         print(f'Loss: {loss.item()}')
#         torch.save(embeddings, embedding_fp)
    
#     # ------------ inference ------------
#     print("running inference")
#     with torch.no_grad():
#         embed_feat = embeddings(torch.tensor(0).to(device)).unsqueeze(0)
#         # predict occ to be turned into grid
#         pred_occ_grid = model.get_occ(query_points, embed_feat)
#         # predict occ for flexi
#         pred_occ = model.get_occ(transformed_points, embed_feat)
#         sdf_deform = model.get_sdf_deform(x_nx3, embed_feat)
#         _, _, sdf, vertices, faces = run_flexi(
#             0, 0, sdf_deform, pred_occ, gt_meshes, test=True)

#     # ------------ results paths ------------
#     from utils import visualize    
#     gt_color = [31, 119, 180, 255]
#     mesh_pred_path = os.path.join(results_dir, 'mesh_pred.png')
#     mesh_flexi_path = os.path.join(results_dir, 'mesh_flexi.png')
#     mesh_gt_path = os.path.join(results_dir, 'mesh_gt.png')
#     lst_paths = [
#         mesh_gt_path,
#         mesh_pred_path,
#         mesh_flexi_path]

#     mag = 0.8; white_bg = False

#     # ------------ gt ------------
#     print("visualizing gt mesh")
#     obj_dir = os.path.join(partnet_dir, anno_id, 'vox_models')
#     assert os.path.exists(obj_dir)
#     gt_mesh_path = os.path.join(obj_dir, f'{anno_id}.obj')
#     gt_mesh = trimesh.load(gt_mesh_path, file_type='obj', force='mesh')
#     gt_vertices_aligned = torch.from_numpy(
#         gt_mesh.vertices).to(device).to(torch.float32)
#     gt_vertices_aligned = kaolin.ops.pointcloud.center_points(
#         gt_vertices_aligned.unsqueeze(0), normalize=True).squeeze(0)
#     gt_vertices_aligned = gt_vertices_aligned.cpu().numpy()
#     aligned_gt_mesh = trimesh.Trimesh(gt_vertices_aligned, gt_mesh.faces)
#     aligned_gt_mesh.export(os.path.join(results_dir, 
#                                         f'{anno_id}_mesh_gt_{expt_id}.obj'))
#     aligned_gt_mesh.visual.vertex_colors = gt_color
#     gt_mesh_vis = visualize.save_mesh_vis(aligned_gt_mesh, mesh_gt_path,
#                                           mag=mag, white_bg=white_bg,
#                                           save_img=False)
    
#     # ------------ occ ------------
#     print("visualizing pred mesh")
#     occ_grid = torch.reshape(
#         pred_occ_grid,
#         (1, pt_sample_res, pt_sample_res, pt_sample_res))
#     occ_grid = torch.permute(occ_grid, (0, 2, 1, 3))
#     pred_vertices, pred_faces =\
#         kaolin.ops.conversions.voxelgrids_to_trianglemeshes(occ_grid)
#     pred_vertices = kaolin.ops.pointcloud.center_points(
#         pred_vertices[0].unsqueeze(0), normalize=True).squeeze(0)
#     pred_vertices = pred_vertices.cpu().numpy()
#     pred_faces = pred_faces[0].cpu().numpy()
#     pred_mesh = trimesh.Trimesh(pred_vertices, pred_faces)
#     pred_mesh.export(os.path.join(results_dir, 
#                                   f'{anno_id}_mesh_pred_{expt_id}.obj'))
#     pred_mesh_vis = visualize.save_mesh_vis(pred_mesh, mesh_pred_path,
#                                             mag=mag, white_bg=white_bg,
#                                             save_img=False)

#     # ------------ flexi ------------
#     print("visualizing flexi mesh")
#     flexi_faces = faces.detach().cpu().numpy()
#     flexi_vertices_aligned = kaolin.ops.pointcloud.center_points(
#         vertices.detach().unsqueeze(0), normalize=True).squeeze(0)
#     flexi_vertices_aligned = flexi_vertices_aligned.cpu().numpy()
#     flexi_mesh = trimesh.Trimesh(flexi_vertices_aligned, flexi_faces)
#     flexi_mesh.export(os.path.join(results_dir,
#                                    f'{anno_id}_flexi_mesh_{expt_id}.obj'))
#     flexi_mesh_vis = visualize.save_mesh_vis(flexi_mesh, mesh_flexi_path,
#                                              mag=mag, white_bg=white_bg,
#                                              save_img=False)
#     print("saving flexi sdf")
#     np.save(os.path.join(results_dir, 
#                          f'{anno_id}_flexi_sdf_{expt_id}.npy'),
#             sdf.detach().cpu().numpy())
    
#     print("saving images")
#     visualize.stitch_imges(
#         os.path.join(results_dir,f'{anno_id}_results_{expt_id}.png'),
#         images=[gt_mesh_vis, pred_mesh_vis, flexi_mesh_vis],
#         adj=100)