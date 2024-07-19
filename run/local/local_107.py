import os
import time
import json
import h5py
import torch
import kaolin
import trimesh
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from occ_networks.basic_decoder_nasa import SDFDecoder, get_embedder
from occ_networks.xform_decoder_nasa_obbgnn_ae_97 import SDFDecoder, get_embedder
from utils import misc, reconstruct, tree
# from utils import visualize
from data_prep import preprocess_data_17
from typing import Dict, List
from anytree.exporter import UniqueDotExporter
from sklearn.decomposition import PCA
from flexi.flexicubes import FlexiCubes
# from flexi.util import *
from flexi import render, util


# inherited checkpoint from local_66

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str)
parser.add_argument('--train', action="store_true")
parser.add_argument('--test', action="store_true")
parser.add_argument('--test_idx', '--ti', type=int)
parser.add_argument('--mask', action="store_true")
parser.add_argument('--recon_one_part', '--ro', action="store_true")
parser.add_argument('--recon_the_rest', '--rt', action="store_true")
# parser.add_argument('--part', type=int)
parser.add_argument('--parts', '--p', nargs='+')
parser.add_argument('--it', type=int)
parser.add_argument('--asb', action="store_true")
parser.add_argument('--asb_scaling', action="store_true")
parser.add_argument('--of', action="store_true", default=False)
parser.add_argument('--of_idx', '--oi', type=int)
parser.add_argument('--inv', action="store_true")
parser.add_argument('--part_indices', nargs='+')
parser.add_argument('--samp', action="store_true")
parser.add_argument('--sample_idx', '--si', type=int)
parser.add_argument('--shape_complete', '--sc', action="store_true")
parser.add_argument('--fixed_indices', nargs='+')
parser.add_argument('--post_process', '--pp', action="store_true")  # with dmtet
parser.add_argument('--post_process_fc', '--ppfc', action="store_true") # with flexicubes
args = parser.parse_args()
# assert args.train != args.test, "Must pass in either train or test"
if args.test:
    assert args.it != None, "Must pass in ckpt iteration if testing"
if args.asb:
    assert args.it != None, "Must pass in ckpt iteration if synthesizing"
if args.mask:
    assert args.recon_one_part != args.recon_the_rest,\
        "Either reconstruct one part or the rest when testing with mask"
    assert args.parts is not None, "Must supply a part"
if args.of:
    assert args.of_idx != None

device = 'cuda'
# lr = 5e-3
lr = 0.01
laplacian_weight = 0.1
iterations = 10001
save_every = 100
multires = 2
pt_sample_res = 64        # point_sampling

expt_id = 107

OVERFIT = args.of
overfit_idx = args.of_idx

batch_size = 1
if OVERFIT:
    batch_size = 1

ds_start = 0
# ds_end = 3272
ds_end = 100

shapenet_dir = '/datasets/ShapeNetCore'
partnet_dir = '/datasets/PartNet'
alignment_dir = 'transmats'
partnet_index_path = '/sota/partnet_dataset/stats/all_valid_anno_info.txt'
name_to_cat = {
    'Chair': '03001627',
    'Lamp': '03636649'
}
cat_name = 'Chair'
cat_id = name_to_cat[cat_name]

train_new_ids_to_objs_path = f'data/{cat_name}_train_new_ids_to_objs_17_{ds_start}_{ds_end}.json'
with open(train_new_ids_to_objs_path, 'r') as f:
    train_new_ids_to_objs: Dict = json.load(f)
model_idx_to_anno_id = {}
for model_idx, anno_id in enumerate(train_new_ids_to_objs.keys()):
    model_idx_to_anno_id[model_idx] = anno_id
with open('results/mapping.json', 'w') as f:
    json.dump(model_idx_to_anno_id, f)

train_data_path = f'data/{cat_name}_train_{pt_sample_res}_17_{ds_start}_{ds_end}.hdf5'
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

# {
#     "chair_arm": 0,
#     "chair_back": 1,
#     "chair_seat": 2,
#     "regular_leg_base": 3
# }

connectivity = [[0, 1], [0, 2], [1, 2], [2, 3]]
connectivity = torch.tensor(connectivity, dtype=torch.long).to(device)

num_union_nodes = adj.shape[1]
num_points = normalized_points.shape[1]
num_shapes, num_parts = part_num_indices.shape
all_fg_part_indices = []
for i in range(num_shapes):
    indices = preprocess_data_17.convert_flat_list_to_fg_part_indices(
        part_num_indices[i], all_indices[i])
    all_fg_part_indices.append(np.array(indices, dtype=object))

num_parts = 1

# n_batches = num_shapes // batch_size
n_batches = 1
# n_batches = 128

if not OVERFIT:
    logs_path = os.path.join('logs', f'local_{expt_id}-bs-{batch_size}',
                             f'{pt_sample_res}')
    ckpt_dir = os.path.join(logs_path, 'ckpt')
    results_dir = os.path.join('results', f'local_{expt_id}-bs-{batch_size}',
                               f'{pt_sample_res}')
else:
    logs_path = os.path.join('logs',
                             f'local_{expt_id}-bs-{batch_size}-of-{overfit_idx}',
                             f'{pt_sample_res}')
    ckpt_dir = os.path.join(logs_path, 'ckpt')
    results_dir = os.path.join('results',
                               f'local_{expt_id}-bs-{batch_size}-of-{overfit_idx}',
                               f'{pt_sample_res}')
misc.check_dir(ckpt_dir)
misc.check_dir(results_dir)
print("results dir: ", results_dir)

writer = SummaryWriter(os.path.join(logs_path, 'summary'))

# -------- model --------
each_part_feat = 128
model = SDFDecoder(num_parts=num_parts,
                   feature_dims=each_part_feat,
                   internal_dims=128,
                   hidden=4,
                   multires=2).to(device)

mse = torch.nn.MSELoss()

params = [p for _, p in model.named_parameters()]

import math
embeddings = torch.nn.Embedding(num_shapes,
                                num_parts*each_part_feat).to(device)
if args.train:
    torch.nn.init.normal_(
        embeddings.weight.data,
        0.0,
        1 / math.sqrt(num_parts*each_part_feat),
    )

fc_voxel_grid_res = 31
# train_res = [256, 256]
train_res = [512, 512]
fc = FlexiCubes(device)
fc_voxel_grid_res = 31
x_nx3, cube_fx8 = fc.construct_voxel_grid(fc_voxel_grid_res)
x_nx3 = 2*x_nx3

# x_nx3 = 2*x_nx3_orig # scale up the grid so that it's larger than the target object
# x_nx3_orig = x_nx3_orig.unsqueeze(0).expand(batch_size, -1, -1)

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
        # return util.load_mesh(gt_mesh_path, device)
        return util.load_mesh_vf(gt_vertices_aligned, gt_mesh.faces, device)
    else:
        return trimesh.load_mesh(gt_mesh_path)

def load_batch(batch_idx, batch_size):
    start = batch_idx*batch_size
    end = start + batch_size
    return all_fg_part_indices[start:end],\
        torch.from_numpy(normalized_points[start:end]).to(device, torch.float32),\
        torch.from_numpy(values[start:end]).to(device, torch.float32),\
        torch.from_numpy(occ[start:end]).to(device, torch.float32),\
        embeddings(torch.arange(start, end).to(device)),\
        torch.from_numpy(node_features[start:end, :, :3]).to(device, torch.float32),\
        torch.from_numpy(adj[start:end]).to(device, torch.long),\
        torch.from_numpy(part_nodes[start:end]).to(device, torch.long),\
        torch.from_numpy(xforms[start:end, :, :3, 3]).to(device, torch.float32),\
        torch.from_numpy(relations[start:end, :, :3, 3]).to(device, torch.float32)

sdf = torch.rand_like(x_nx3[:,0]) - 0.1 # randomly init SDF
sdf    = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
# weight = torch.zeros((cube_fx8.shape[0], 21), dtype=torch.float, device='cuda') 
# weight    = torch.nn.Parameter(weight.clone().detach(), requires_grad=True)
deform = torch.nn.Parameter(torch.zeros_like(x_nx3), requires_grad=True)

# optimizer = torch.optim.Adam([{"params": params, "lr": lr},
#                             #   {"params": embeddings.parameters(), "lr": lr}
#                               ])
optimizer = torch.optim.Adam([sdf, deform])
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda x: max(0.0, 10**(-x*0.0002)))

timelapse_dir = os.path.join(results_dir, 'training_timelapse')
print("timelapse dir: ", timelapse_dir)
timelapse = kaolin.visualize.Timelapse(timelapse_dir)


def loss_f(pred_values, gt_values):
    recon_loss = mse(pred_values, gt_values)
    return recon_loss


def loss_f_xform(pred_xforms, gt_xforms):
    xform_loss = mse(pred_xforms, gt_xforms)
    return xform_loss


torch.manual_seed(319)
np.random.seed(319)

embed_fn, _ = get_embedder(2)

# model.pre_train_sphere(1000)

# from dmtet_networks import dmtet_utils
# grid_res = 128
# tet_verts = torch.tensor(
#     np.load('samples/{}_verts.npz'.format(grid_res))['data'],
#     dtype=torch.float, device=device)
# tets = torch.tensor(
#     ([np.load('samples/{}_tets_{}.npz'.format(grid_res, i))['data'] for i in range(4)]),
#     dtype=torch.long, device=device).permute(1,0)

# pcd, _ = trimesh.sample.sample_surface(my_load_mesh(0), 10000, seed=319)
# pcd = torch.from_numpy(pcd).to(device, torch.float32)

gt_mesh = my_load_mesh(2)

def train_one_itr(it, b,
                  all_fg_part_indices, 
                  batch_points, batch_values, batch_occ,
                  batch_embed,
                  batch_node_feat, batch_adj, batch_part_nodes,
                  batch_xforms, batch_relations):
    optimizer.zero_grad()

    # dmtet_out = model.get_sdf_deform(tet_verts, None)
    # # print(dmtet_out.shape)
    # # sdf, deform = dmtet_out[:, :, :1], dmtet_out[:, :, 1:]
    # sdf, deform = dmtet_out[:, :1], dmtet_out[:, 1:]
    # verts_deformed = tet_verts + torch.tanh(deform) / grid_res # constraint deformation to avoid flipping tets
    # mesh_verts, mesh_faces = kaolin.ops.conversions.marching_tetrahedra(
    #     verts_deformed.unsqueeze(0), tets, sdf.unsqueeze(0)) # running MT (batched) to extract surface mesh
    # mesh_verts, mesh_faces = mesh_verts[0], mesh_faces[0]

    # loss = dmtet_utils.loss_f(iterations, mesh_verts, mesh_faces, pcd, it)

    # # flexi_out = model.get_sdf_deform(batch_points, batch_embed)
    # flexi_out = model.get_sdf_deform(batch_points, None)

    # # pred_sdf = -2*pred_values2 + 1 + flexi_out[:, :, :1]
    # pred_sdf = flexi_out[:, :, :1]
    # deform = flexi_out[:, :, 1:]

    # sdf_loss = loss_f(pred_sdf, batch_values)

    # if it >= 0:
    #     mesh_loss = 0
    #     for ele in range(batch_size):
    # gt_mesh = my_load_mesh(batch_size*b + ele)
    mv, mvp = render.get_random_camera_batch(
        8, iter_res=train_res, device=device, use_kaolin=False)
    target = render.render_mesh_paper(gt_mesh, mv, mvp, train_res)

    grid_verts = x_nx3 + (2-1e-8) / (fc_voxel_grid_res * 2) * torch.tanh(
        # deform[ele])
        deform)
    # mod_sdf = torch.squeeze(pred_sdf[ele])
    # mod_sdf = torch.squeeze(sdf)
    vertices, faces, L_dev = fc(
        grid_verts, sdf,
        cube_fx8, fc_voxel_grid_res, training=True)

    flexicubes_mesh = util.Mesh(vertices, faces)

    try: 
        buffers = render.render_mesh_paper(flexicubes_mesh, mv, mvp, train_res)
    except Exception as e:
        import logging, traceback
        logging.error(traceback.format_exc())
        # print(torch.min(pred_sdf[ele]))
        # print(torch.max(pred_sdf[ele]))
        print(torch.min(sdf))
        print(torch.max(sdf))

    mask_loss = (buffers['mask'] - target['mask']).abs().mean()
    depth_loss = (((((buffers['depth'] - (target['depth']))* target['mask'])**2).sum(-1)+1e-8)).sqrt().mean() * 10

    mesh_loss = mask_loss + depth_loss
        # mesh_loss /= batch_size

        # loss = mesh_loss + sdf_loss
        # loss = mesh_loss
    # else:
    #     loss = sdf_loss

    # if b == n_batches - 1:
    #     # writer.add_scalar('occ loss', loss2, it)
    #     # writer.add_scalar('sdf loss', sdf_loss, it)
    #     if it >= 0:
    #         writer.add_scalar('mesh loss', mesh_loss, it)

    if it != 0 and (it) % 100 == 0 or it == (iterations - 1): 
        print ('Iteration {} - loss: {:.5f}, # of mesh vertices: {}, # of mesh faces: {}'.format(it, loss, vertices.shape[0], faces.shape[0]))
        timelapse.add_mesh_batch(
            iteration=it,
            category='extracted_mesh',
            vertices_list=[vertices.cpu()],
            faces_list=[faces.cpu()]
        )
        timelapse.add_mesh_batch(
            iteration=it,
            category='gt_mesh',
            vertices_list=[torch.from_numpy(np.array(my_load_mesh(2, True).vertices))],
            faces_list=[torch.from_numpy(np.array(my_load_mesh(2, True).faces))]
        )

    mesh_loss.backward()
    optimizer.step()
    scheduler.step()

    return mesh_loss


if args.train:
    print("training...")
    start_time = time.time()
    for it in range(iterations):
        batch_loss = 0
        for b in range(n_batches):
            batch_fg_part_indices,\
                batch_normalized_points, batch_values, batch_occ,\
                batch_embed, \
                batch_node_feat, batch_adj, batch_part_nodes,\
                batch_xforms, batch_relations =\
                    load_batch(b, batch_size)
            loss = train_one_itr(it, b,
                                 batch_fg_part_indices,
                                 batch_normalized_points,
                                 batch_values,
                                 batch_occ,
                                 batch_embed,
                                 batch_node_feat, batch_adj, batch_part_nodes,
                                 batch_xforms, batch_relations)
            writer.add_scalar('iteration loss', loss, it)
            batch_loss += loss
        avg_batch_loss = batch_loss / batch_size
            
        show = 100 if OVERFIT else 10
        if (it) % show == 0 or it == (iterations - 1):
            info = f'-------- Iteration {it} - loss: {avg_batch_loss:.8f} --------'
            print(info)

        save = 1000 if OVERFIT else 200
        if (it) % save == 0 or it == (iterations - 1):
            torch.save({
                'epoch': it,
                'model_state_dict': model.state_dict(),
                'embeddings_state_dict': embeddings.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_batch_loss': avg_batch_loss,
                }, os.path.join(ckpt_dir, f'model_{it}.pt'))

    print("duration: ", time.time() - start_time)
    writer.close()


def compute_iou(voxel_grid1, voxel_grid2):
    assert voxel_grid1.shape == voxel_grid2.shape,\
        "Voxel grids must have the same shape."
    intersection = torch.logical_and(voxel_grid1, voxel_grid2).sum()
    union = torch.logical_or(voxel_grid1, voxel_grid2).sum()
    iou = intersection / float(union) if union != 0 else 1.0
    return iou


if args.test:
    from utils import visualize

    white_bg = True
    it = args.it
    if not OVERFIT:
        model_idx = args.test_idx
    else:
        model_idx = overfit_idx
    if not OVERFIT:
        anno_id = model_idx_to_anno_id[model_idx]
    else:
        anno_id = model_idx_to_anno_id[model_idx]
    model_id = misc.anno_id_to_model_id(partnet_index_path)[anno_id]
    print(f"anno id: {anno_id}, model id: {model_id}")

    checkpoint = torch.load(os.path.join(ckpt_dir, f'model_{it}.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    embeddings = torch.nn.Embedding(num_shapes, num_parts*each_part_feat).to(device)
    embeddings.load_state_dict(checkpoint['embeddings_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    results_dir = os.path.join(results_dir, anno_id)
    misc.check_dir(results_dir)

    if not OVERFIT:
        # part_nodes = part_nodes[model_idx].unsqueeze(0)
        batch_embed = embeddings(torch.tensor(model_idx).to(device)).unsqueeze(0)
    else:
        batch_embed = embeddings(torch.tensor(0).to(device)).unsqueeze(0)

    unique_part_names, name_to_ori_ids_and_objs,\
        orig_obbs, entire_mesh, name_to_obbs, _, _, _ =\
        preprocess_data_17.merge_partnet_after_merging(anno_id)

    with open(f'data/{cat_name}_part_name_to_new_id_17_{ds_start}_{ds_end}.json', 'r') as f:
        unique_name_to_new_id = json.load(f)

    part_obbs = []

    unique_names = list(unique_name_to_new_id.keys())
    model_part_names = list(name_to_obbs.keys())

    existing_parts = []

    for i, un in enumerate(unique_names):
        if not un in model_part_names:
            part_obbs.append([])
            continue
        part_obbs.append([name_to_obbs[un]])
        existing_parts.append(i)

    gt_color = [31, 119, 180, 255]
    mesh_gt_path = os.path.join(results_dir, 'mesh_gt.png')
    flexi_path = os.path.join(results_dir, 'mesh_flexi.png')
    lst_paths = [
        mesh_gt_path,
        flexi_path]

    # query_points = reconstruct.make_query_points(pt_sample_res)
    # query_points = torch.from_numpy(query_points).to(device, torch.float32)
    # query_points = query_points.unsqueeze(0)
    x_nx3, cube_fx8 = fc.construct_voxel_grid(31)
    query_points = x_nx3.unsqueeze(0) * 1.1
    bs, num_points, _ = query_points.shape

    if not OVERFIT:
        batch_node_feat = torch.from_numpy(node_features[model_idx:model_idx+1, :, :3]).to(device, torch.float32)
        batch_adj = torch.from_numpy(adj[model_idx:model_idx+1]).to(device, torch.float32)
        batch_part_nodes = torch.from_numpy(part_nodes[model_idx:model_idx+1]).to(device, torch.float32)
    else:
        batch_node_feat = torch.from_numpy(node_features[:, :, :3]).to(device, torch.float32)
        batch_adj = torch.from_numpy(adj).to(device, torch.float32)
        batch_part_nodes = torch.from_numpy(part_nodes).to(device, torch.float32)

    # print(batch_node_feat.shape)
    # print(batch_adj.shape)
    # print(batch_part_nodes.shape)

    # with torch.no_grad():
    
    #     transformed_points = query_points.unsqueeze(1).expand(-1, num_parts, -1, -1)

    #     occs1 = model(transformed_points, batch_embed)
    #     pred_values, _ = torch.max(occs1, dim=-1, keepdim=True)

    if args.mask:
        mag = 1
    else:
        mag = 0.8

    # np.save(os.path.join(results_dir, f'{anno_id}_occ.npy'),
    #         torch.squeeze(pred_values).detach().cpu().numpy())

    with torch.no_grad():
        flexi_out = model.get_sdf_deform(query_points, batch_embed)
    # pred_sdf = -2*pred_values + 1 + flexi_out[:, :, :1]
    pred_sdf = flexi_out[:, :, :1]

    # np.save(os.path.join(results_dir, f'{anno_id}_sdf.npy'),
    #         torch.squeeze(pred_sdf).detach().cpu().numpy())
    
    deform = flexi_out[:, :, 1:]
    grid_verts = x_nx3 + (2-1e-8) / (fc_voxel_grid_res * 2) * torch.tanh(
        deform[0])
    mod_sdf = torch.squeeze(pred_sdf[0])
    vertices, faces, L_dev = fc(
        grid_verts, mod_sdf,
        cube_fx8, fc_voxel_grid_res, training=True)
    # flexicubes_mesh = util.Mesh(vertices, faces)
    flexi_mesh = trimesh.Trimesh(vertices = vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), process=False)
    flexi_mesh.export(os.path.join(results_dir, 'mesh_flexi.obj'))
    visualize.save_mesh_vis(flexi_mesh, flexi_path,
                            mag=mag, white_bg=white_bg)

    pt_sample_res = 32
    
    unmasked_indices = list(set(range(num_parts)) - set([]))

    obj_dir = os.path.join(partnet_dir, anno_id, 'vox_models')
    assert os.path.exists(obj_dir)
    gt_mesh_path = os.path.join(obj_dir, f'{anno_id}.obj')

    if not args.mask:
        gt_mesh = trimesh.load(gt_mesh_path, file_type='obj', force='mesh')
        gt_vertices_aligned = torch.from_numpy(
            gt_mesh.vertices).to(device).to(torch.float32)
        gt_vertices_aligned = kaolin.ops.pointcloud.center_points(
            gt_vertices_aligned.unsqueeze(0), normalize=True).squeeze(0)
        gt_vertices_aligned = gt_vertices_aligned.cpu().numpy()
        aligned_gt_mesh = trimesh.Trimesh(gt_vertices_aligned, gt_mesh.faces)
        aligned_gt_mesh.export(os.path.join(results_dir, 'mesh_gt.obj'))
        aligned_gt_mesh.visual.vertex_colors = [31, 119, 180]
        visualize.save_mesh_vis(aligned_gt_mesh, mesh_gt_path,
                                mag=mag, white_bg=white_bg)
    else:
        partnet_objs_dir = os.path.join(partnet_dir, anno_id, 'objs')
        lst_of_part_meshes = []
        
        model_new_ids_to_obj_names: Dict = train_new_ids_to_objs[anno_id]
        for i in range(num_parts):
            if not str(i) in list(model_new_ids_to_obj_names.keys()):
                lst_of_part_meshes.append(None)
            else:
                obj_names = model_new_ids_to_obj_names[str(i)]
                meshes = []
                for obj_name in obj_names:
                    obj_path = os.path.join(partnet_objs_dir, f'{obj_name}.obj')
                    part_mesh = trimesh.load_mesh(obj_path)
                    meshes.append(part_mesh)
                concat_part_mesh = trimesh.util.concatenate(meshes)
                lst_of_part_meshes.append(concat_part_mesh)

        masked_indices = []
        unmasked_indices = list(set(range(num_parts)) - set(masked_indices))

        concat_mesh = trimesh.util.concatenate(
            [lst_of_part_meshes[x] for x in unmasked_indices])
    
        aligned_verts = kaolin.ops.pointcloud.center_points(
            torch.from_numpy(np.array(concat_mesh.vertices)).unsqueeze(0),
            normalize=True).squeeze(0)
        aligned_gt_mesh = trimesh.Trimesh(aligned_verts, concat_mesh.faces)
        aligned_gt_mesh.export(os.path.join(results_dir, 'mesh_gt.obj'))
        aligned_gt_mesh.visual.vertex_colors = [31, 119, 180]
        visualize.save_mesh_vis(aligned_gt_mesh, mesh_gt_path,
                                mag=mag, white_bg=white_bg)

    gt_samples, _ = trimesh.sample.sample_surface(
        aligned_gt_mesh, 10000, seed=319)
    gt_samples = torch.from_numpy(gt_samples).to(device)
    pred_samples, _ = trimesh.sample.sample_surface(
        flexi_mesh, 10000, seed=319)
    pred_samples = torch.from_numpy(pred_samples).to(device)
    chamfer = kaolin.metrics.pointcloud.chamfer_distance(
        pred_samples.unsqueeze(0), gt_samples.unsqueeze(0)).mean()
    print("[EVAL] Chamfer distance: ", chamfer.cpu().numpy())
    
    if not args.mask:
        visualize.stitch_imges(
            os.path.join(results_dir,f'{anno_id}_results.png'),
            image_paths=lst_paths,
            adj=100)
    else:
        # parts_str contains all the parts that are MASKED OUT
        parts_str = '-'.join([str(x) for x in masked_indices])
        visualize.stitch_imges(
            os.path.join(results_dir,f'{anno_id}_results_mask_{parts_str}.png'),
            image_paths=lst_paths,
            adj=100)

    exit(0)

