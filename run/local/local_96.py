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
from occ_networks.xform_decoder_nasa_obbgnn_ae_96 import SDFDecoder, get_embedder
from utils import misc, reconstruct, tree
# from utils import visualize
from data_prep import preprocess_data_17
from typing import Dict, List
from anytree.exporter import UniqueDotExporter
from sklearn.decomposition import PCA
from flexi.flexicubes import FlexiCubes
# from flexi.util import *
from flexi import render, loss, util


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

expt_id = 96

OVERFIT = args.of
overfit_idx = args.of_idx

batch_size = 25
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
each_part_feat = 32
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
train_res = [256, 256]
fc = FlexiCubes(device)
fc_voxel_grid_res = 31
x_nx3, cube_fx8 = fc.construct_voxel_grid(fc_voxel_grid_res)
x_nx3 = 2*x_nx3
# x_nx3 = 2*x_nx3_orig # scale up the grid so that it's larger than the target object
# x_nx3_orig = x_nx3_orig.unsqueeze(0).expand(batch_size, -1, -1)

def my_load_mesh(model_idx):
    anno_id = model_idx_to_anno_id[model_idx]
    obj_dir = os.path.join(partnet_dir, anno_id, 'vox_models')
    assert os.path.exists(obj_dir)
    gt_mesh_path = os.path.join(obj_dir, f'{anno_id}.obj')
    return util.load_mesh(gt_mesh_path, device)

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

optimizer = torch.optim.Adam([{"params": params, "lr": lr},
                              {"params": embeddings.parameters(), "lr": lr}])
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda x: max(0.0, 10**(-x*0.0002)))


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

def train_one_itr(it, b,
                  all_fg_part_indices, 
                  batch_points, batch_values, batch_occ,
                  batch_embed,
                  batch_node_feat, batch_adj, batch_part_nodes,
                  batch_xforms, batch_relations):
    optimizer.zero_grad()

    num_parts_to_mask = np.random.randint(1, num_parts)
    # num_parts_to_mask = 1
    rand_indices = np.random.choice(num_parts, num_parts_to_mask,
                                    replace=False)

    masked_indices = torch.from_numpy(rand_indices).to(device, torch.long)
    # make gt value mask
    val_mask = torch.ones_like(batch_occ).to(device, torch.float32)
    for i in range(batch_size):
        fg_part_indices = all_fg_part_indices[i]
        fg_part_indices_masked = fg_part_indices[masked_indices.cpu().numpy()]
        if len(fg_part_indices_masked) != 0:
            fg_part_indices_masked = np.concatenate(fg_part_indices_masked, axis=0)
        else:
            fg_part_indices_masked = np.array([])
        val_mask[i][fg_part_indices_masked] = 0
    modified_values = batch_occ * val_mask

    parts_mask = torch.zeros((batch_embed.shape[0], num_parts)).to(device, torch.float32)
    if len(masked_indices) != 0:
        parts_mask[:, rand_indices] = 1
    parts_mask = torch.repeat_interleave(parts_mask,
                                         each_part_feat, dim=-1)

    points_mask = torch.zeros((batch_size, num_parts, 1, 1)).to(device, torch.float32)
    points_mask[:, rand_indices] = 1

    occ_mask = torch.zeros((batch_size, 1, num_parts)).to(device, torch.float32)
    occ_mask[:, :, rand_indices] = 1

    # learning xforms
    batch_vec = torch.arange(start=0, end=batch_size).to(device)
    batch_vec = torch.repeat_interleave(batch_vec, num_union_nodes)
    batch_mask = torch.sum(batch_part_nodes, dim=1)
    learned_geom, learned_xforms = model.learn_geom_xform(batch_node_feat,
                                            batch_adj,
                                            batch_mask,
                                            batch_vec)
    learned_relations = learned_xforms[:, connectivity[:, 0], connectivity[:, 1], :]
    learned_xforms = learned_xforms[:, [0, 1, 2, 3], [0, 1, 2, 3], :]

    batch_geom = torch.einsum('ijk, ikm -> ijm',
                               batch_part_nodes.to(torch.float32),
                               batch_node_feat)

    pairwise_xforms = learned_xforms[:, connectivity]
    learned_relations = pairwise_xforms[:, :, 1] - pairwise_xforms[:, :, 0]

    transformed_points = batch_points.unsqueeze(1).expand(-1, num_parts, -1, -1) +\
        learned_xforms.unsqueeze(2)

    occs1 = model(transformed_points.masked_fill(points_mask==1,torch.tensor(0)),
                  batch_embed.masked_fill(parts_mask==1, torch.tensor(0)))
    occs1 = occs1.masked_fill(occ_mask==1,torch.tensor(float('-inf')))
    pred_values1, _ = torch.max(occs1, dim=-1, keepdim=True)
    # pred_values1, _ = torch.min(occs1, dim=-1, keepdim=True)
    # pred_values1_occ = (pred_values1 < 0).int()
    loss1 = loss_f(pred_values1, modified_values)
    # loss1 = loss_f(pred_values1_occ, modified_values)

    occs2 = model(transformed_points,
                  batch_embed)
    pred_values2, _ = torch.max(occs2, dim=-1, keepdim=True)
    # pred_values2, _ = torch.min(occs2, dim=-1, keepdim=True)
    # pred_values2_occ = (pred_values2 < 0).int()
    loss2 = loss_f(pred_values2, batch_occ)
    # loss2 = loss_f(pred_values2_occ, batch_occ)

    loss = loss1 + loss2

    loss_bbox_geom = loss_f(
        embed_fn(learned_geom.view(-1, 3)),
        embed_fn(batch_geom.view(-1, 3)),)
    loss += loss_bbox_geom

    loss_xform = loss_f_xform(
        embed_fn(learned_xforms.view(-1, 3)),
        embed_fn(batch_xforms.view(-1, 3)))
    loss_relations = loss_f_xform(
        embed_fn(learned_relations.view(-1, 3)),
        embed_fn(batch_relations.view(-1, 3)))
    
    loss += 10 * loss_xform + 10 * loss_relations

    flexi_out = model.get_sdf_deform(batch_points, batch_embed)

    pred_sdf = -2*pred_values2 + 1 + flexi_out[:, :, :1]
    # pred_sdf = flexi_out[:, :, :1]
    deform = flexi_out[:, :, 1:]

    sdf_loss = loss_f(pred_sdf, batch_values)
    
    if it >= 1000:
        mesh_loss = 0
        for ele in range(batch_size):
            gt_mesh = my_load_mesh(batch_size*b + ele)
            mv, mvp = render.get_random_camera_batch(
                8, iter_res=train_res, device=device, use_kaolin=False)
            target = render.render_mesh_paper(gt_mesh, mv, mvp, train_res)

            grid_verts = x_nx3 + (2-1e-8) / (fc_voxel_grid_res * 2) * torch.tanh(
                deform[ele])
            mod_sdf = torch.squeeze(pred_sdf[ele])
            vertices, faces, L_dev = fc(
                grid_verts, mod_sdf,
                cube_fx8, fc_voxel_grid_res, training=True)

            flexicubes_mesh = util.Mesh(vertices, faces)

            try: 
                buffers = render.render_mesh_paper(flexicubes_mesh, mv, mvp, train_res)
            except Exception as e:
                import logging, traceback
                logging.error(traceback.format_exc())
                print(torch.min(pred_sdf[ele]))
                print(torch.max(pred_sdf[ele]))

            mask_loss = (buffers['mask'] - target['mask']).abs().mean()
            depth_loss = (((((buffers['depth'] - (target['depth']))* target['mask'])**2).sum(-1)+1e-8)).sqrt().mean() * 10

            mesh_loss += mask_loss + depth_loss
        mesh_loss /= batch_size

        loss += 0.1 * mesh_loss + sdf_loss
    else:
        loss += sdf_loss

    if b == n_batches - 1:
        writer.add_scalar('occ loss', loss1 + loss2, it)
        writer.add_scalar('bbox geom loss', loss_bbox_geom, it)
        writer.add_scalar('xform loss', loss_xform, it)
        writer.add_scalar('relations loss', loss_relations, it)
        writer.add_scalar('sdf loss', sdf_loss, it)
        if it >= 1000:
            writer.add_scalar('mesh loss', mesh_loss, it)

    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss


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

        save = 1000 if OVERFIT else 100
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
    mesh_pred_path = os.path.join(results_dir, 'mesh_pred.png')
    mesh_gt_path = os.path.join(results_dir, 'mesh_gt.png')
    learned_obbs_path = os.path.join(results_dir, 'obbs_pred.png')
    obbs_path = os.path.join(results_dir, 'obbs_gt.png')
    flexi_path = os.path.join(results_dir, 'mesh_flexi.png')
    lst_paths = [
        obbs_path,
        mesh_gt_path,
        learned_obbs_path,
        mesh_pred_path,
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

    with torch.no_grad():
        if args.mask:
            # parts = [args.part]
            parts = args.parts
            parts = [int(x) for x in parts]
            if args.recon_one_part:
                # only reconstruct part specified by parts
                masked_indices = list(set(range(num_parts)) - set(parts))
                masked_indices = torch.Tensor(masked_indices).to(device, torch.long)
            if args.recon_the_rest:
                # don't reconstruct stuff in masked_indices
                masked_indices = torch.Tensor(parts).to(device, torch.long)
        else:
            masked_indices = torch.Tensor([]).to(device, torch.long)

        if not args.mask:
            print("reconstructing...")
        else:
            print(f"masking parts: {masked_indices.cpu().numpy().tolist()}")

        parts_mask = torch.zeros((batch_embed.shape[0], num_parts)).to(device, torch.float32)
        parts_mask[:, masked_indices] = 1
        parts_mask = torch.repeat_interleave(parts_mask, each_part_feat, dim=-1)
        
        points_mask = torch.zeros((1, num_parts, 1, 1)).to(device, torch.float32)
        points_mask[:, masked_indices] = 1

        occ_mask = torch.zeros((1, 1, num_parts)).to(device, torch.float32)
        occ_mask[:, :, masked_indices] = 1

        # learning xforms
        batch_vec = torch.arange(start=0, end=1).to(device)
        batch_vec = torch.repeat_interleave(batch_vec, num_union_nodes)
        batch_mask = torch.sum(batch_part_nodes, dim=1)
        learned_geom, learned_xforms = model.learn_geom_xform(batch_node_feat,
                                                    batch_adj,
                                                    batch_mask,
                                                    batch_vec)
        learned_relations = learned_xforms[:, connectivity[:, 0], connectivity[:, 1], :]
        learned_xforms = learned_xforms[:, [0, 1, 2, 3], [0, 1, 2, 3], :]

        batch_geom = torch.einsum('ijk, ikm -> ijm',
                                batch_part_nodes.to(torch.float32),
                                batch_node_feat)

        pairwise_xforms = learned_xforms[:, connectivity]
        learned_relations = pairwise_xforms[:, :, 1] - pairwise_xforms[:, :, 0]

        transformed_points = query_points.unsqueeze(1).expand(-1, num_parts, -1, -1) +\
            learned_xforms.unsqueeze(2)

        occs1 = model(transformed_points.masked_fill(points_mask==1,torch.tensor(0)),
                      batch_embed.masked_fill(parts_mask==1, torch.tensor(0)))
        occs1 = occs1.masked_fill(occ_mask==1,torch.tensor(float('-inf')))
        pred_values, _ = torch.max(occs1, dim=-1, keepdim=True)

    learned_xforms = learned_xforms[0].cpu().numpy()
    learned_obbs_of_interest = [[]] * num_parts
    for i in range(num_parts):
        if i not in existing_parts:
            continue
        ext = extents[model_idx, i]
        learned_xform = np.eye(4)
        learned_xform[:3, 3] = -learned_xforms[i]
        ext_xform = (ext, learned_xform)
        learned_obbs_of_interest[i] = [ext_xform]

    if args.mask:
        mag = 1
    else:
        mag = 0.8

    flexi_out = model.get_sdf_deform(query_points, batch_embed)
    pred_sdf = -2*pred_values + 1 + flexi_out[:, :, :1]
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

    sdf_grid = torch.reshape(
        pred_values,
        (1, pt_sample_res, pt_sample_res, pt_sample_res))
    # sdf_grid = torch.permute(sdf_grid, (0, 2, 1, 3))
    sdf_grid = torch.permute(sdf_grid, (0, 3, 1, 2))
    vertices, faces =\
        kaolin.ops.conversions.voxelgrids_to_trianglemeshes(sdf_grid)
    vertices = kaolin.ops.pointcloud.center_points(
        vertices[0].unsqueeze(0), normalize=True).squeeze(0)
    pred_vertices = vertices.cpu().numpy()
    pred_faces = faces[0].cpu().numpy()
    pred_mesh = trimesh.Trimesh(pred_vertices, pred_faces)
    pred_mesh.export(os.path.join(results_dir, 'mesh_pred.obj'))
    visualize.save_mesh_vis(pred_mesh, mesh_pred_path,
                            mag=mag, white_bg=white_bg)
    
    unmasked_indices = list(set(range(num_parts)) - set(masked_indices))

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

        masked_indices = masked_indices.cpu().numpy().tolist()
        unmasked_indices = list(set(range(num_parts)) - set(masked_indices))

        if args.recon_one_part:
            # only show stuff in masked_indices
            concat_mesh = trimesh.util.concatenate(
                [lst_of_part_meshes[x] for x in parts])
        if args.recon_the_rest:
            # don't show stuff in masked_indices
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
        pred_mesh, 10000, seed=319)
    pred_samples = torch.from_numpy(pred_samples).to(device)
    chamfer = kaolin.metrics.pointcloud.chamfer_distance(
        pred_samples.unsqueeze(0), gt_samples.unsqueeze(0)).mean()
    print("[EVAL] Chamfer distance: ", chamfer.cpu().numpy())

    import itertools
    obbs_of_interest = [part_obbs[x] for x in unmasked_indices]
    obbs_of_interest = list(itertools.chain(*obbs_of_interest))
    visualize.save_obbs_vis(obbs_of_interest,
                            obbs_path, mag=mag, white_bg=white_bg,
                            unmasked_indices=unmasked_indices)
    
    learned_obbs_of_interest = [learned_obbs_of_interest[x] for x in unmasked_indices]
    learned_obbs_of_interest = list(itertools.chain(*learned_obbs_of_interest))
    visualize.save_obbs_vis(learned_obbs_of_interest,
                            learned_obbs_path, mag=mag, white_bg=white_bg,
                            unmasked_indices=unmasked_indices)
    
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


def build_adj_matrix(col, adj_to_build, adj_to_look_up):
    # find the rows for which col is one
    # this value must exist and must be unique
    # exist: the tree has no islands, so the node must be connected 
    # unique: one child node only has one parent
    if col == 0:
        return 
    row = torch.argwhere(adj_to_look_up[:, col] == 1).flatten()[0]
    adj_to_build[row, col] = 1
    build_adj_matrix(row, adj_to_build, adj_to_look_up)


if args.asb:
    from utils import visualize

    # assembly shape from coarse parts
    it = args.it

    # model_indices = [20, 20, 20, 20]
    # part_indices = [0, 1, 2, 3]

    # model_indices = [0] * 4
    # part_indices = [0, 1, 2, 3]

    # model_indices = [39, 86, 43, 41]
    # part_indices = [0, 1, 2, 3]
    # part_indices = [1, 2, 3, 0]
    # part_indices = [2, 3, 0, 1]
    # part_indices = [3, 0, 1, 2]
    # part_indices = [2, 3, 1, 0]
    
    # model_indices = [3, 0, 1, 4]
    # model_indices = [4, 9, 6, 3]
    # model_indices = [6, 3, 5, 9]
    # model_indices = [7, 4, 0, 0]
    # model_indices = [5, 4, 2, 8]
    # part_indices = [0, 1, 2, 3]

    # model_indices = [500, 501, 506, 505]
    # part_indices = [1, 0, 3, 2]

    # model_indices = [500, 501, 502, 505]
    # part_indices = [3, 0, 2, 1]
    # part_indices = [3, 0, 1, 2]

    # model_indices = [255, 354, 28, 175]
    # model_indices = [343, 185, 370, 258]
    # model_indices = [347, 82, 292, 365]
    # part_indices = [int(x) for x in args.part_indices]

    # {
    #     "chair_arm": 0,
    #     "chair_back": 1,
    #     "chair_seat": 2,
    #     "regular_leg_base": 3
    # }

    # anno_ids = [model_idx_to_anno_id[x] for x in model_indices]
    # # model_id = misc.anno_id_to_model_id(partnet_index_path)[anno_id]
    # # print(f"anno id: {anno_id}, model id: {model_id}")
    # print(anno_ids)

    anno_ids = ['47914', '44979', '2243', '44164']
    # anno_ids = ['37900', '39901', '37121', '2243']
    # anno_ids = ['36685', '38091', '38567', '38816']
    # anno_ids = ['38567', '36685', '2673', '38816']
    part_indices = [0, 1, 2, 3]

    anno_id_to_model_idx = {v: k for k, v in model_idx_to_anno_id.items()}
    model_indices = [anno_id_to_model_idx[x] for x in anno_ids]
    print(model_indices)
    print(anno_ids)

    asb_str = '-'.join([str(x) for x in anno_ids+part_indices])
    old_results_dir = results_dir
    results_dir = os.path.join(results_dir, f'0assembly_{asb_str}')
    misc.check_dir(results_dir)
    print("results dir: ", results_dir)

    with open(os.path.join(results_dir, 'anno_ids.txt'), 'w') as f:
        anno_ids_str = '-'.join([str(x) for x in model_indices])
        part_indices_str = '-'.join([str(x) for x in part_indices])
        # f.write(anno_ids_str)
        f.writelines([anno_ids_str, '\n', part_indices_str])

    checkpoint = torch.load(os.path.join(ckpt_dir, f'model_{it}.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    if not args.inv:
        embeddings = torch.nn.Embedding(num_shapes, num_parts*each_part_feat)
        embeddings.load_state_dict(checkpoint['embeddings_state_dict'])
    else:
        embeddings = torch.zeros(num_shapes, num_parts*each_part_feat)
        all_paths = [os.path.join(old_results_dir, str(x), 'embedding.pth') for x in anno_ids]
        for p in all_paths:
            assert os.path.exists(p), "you haven't run shape inversion for all unseen models"
        all_embeddings = [torch.load(p) for p in all_paths]
        for i in range(len(all_embeddings)):
            embeddings[i] = all_embeddings[i].weight.data[0]

    all_part_num_obbs = []
    # obbs_of_interest = [[]] * num_parts
    # node features just have the extents
    asb_node_feat = torch.zeros((num_union_nodes, 3)).to(torch.float32)
    asb_part_nodes = torch.zeros((num_parts, num_union_nodes)).to(torch.long)
    asb_adj = torch.zeros((num_union_nodes, num_union_nodes)).to(torch.long)
    asb_embed = torch.zeros(1, num_parts*each_part_feat).to(torch.float32)
    gt_xforms = torch.zeros((1, num_parts, 3)).to(torch.float32)
    gt_exts = torch.zeros((num_parts, 3)).to(torch.float32)
    for i, idx in enumerate(model_indices):
        anno_id = model_idx_to_anno_id[idx]
        part_idx = part_indices[i]
        shape_node_feat = torch.from_numpy(node_features[idx, :, :3])    # 7, 3
        shape_part_nodes = torch.from_numpy(part_nodes[idx])  # 4, 7
        shape_adj = torch.from_numpy(adj[idx])
        shape_xform = torch.from_numpy(xforms[idx, part_idx, :3, 3])
        # shape_part_mask = shape_part_nodes[part_idx].unsqueeze(1)    # 7, 1
        # asb_node_features += shape_node_feat * shape_part_mask
        union_node_indices =\
            torch.argwhere(shape_part_nodes[part_idx] == 1).flatten()
        shape_ext = shape_node_feat[union_node_indices]
        asb_node_feat[union_node_indices] = shape_ext
        
        asb_part_nodes[part_idx] = shape_part_nodes[part_idx]
        
        for uni in union_node_indices:
            build_adj_matrix(uni, asb_adj, shape_adj)

        if not args.inv:
            asb_embed[:, part_idx*each_part_feat:(part_idx+1)*each_part_feat] =\
                embeddings(torch.tensor(idx))[part_idx*each_part_feat:(part_idx+1)*each_part_feat]
        else:
            asb_embed[:, part_idx*each_part_feat:(part_idx+1)*each_part_feat] =\
                embeddings[i, part_idx*each_part_feat:(part_idx+1)*each_part_feat]

        gt_xforms[:, part_idx] = shape_xform
        gt_exts[part_idx] = shape_ext

    # build a tree that is a subset of the union tree (dense graph)
    # using the bounding boxes
    node_names = np.load(f'data/{cat_name}_union_node_names_17_{ds_start}_{ds_end}.npy')
    recon_root = tree.recon_tree(asb_adj.numpy(), node_names)
    UniqueDotExporter(recon_root,
                      indent=0,
                      nodenamefunc=lambda node: node.name,
                      nodeattrfunc=lambda node: "shape=box",).to_picture(
                          os.path.join(results_dir, 'asb_tree.png'))

    batch_node_feat = asb_node_feat.unsqueeze(0).to(device)
    batch_part_nodes = asb_part_nodes.unsqueeze(0).to(device)
    batch_adj = asb_adj.unsqueeze(0).to(device)
    batch_embed = asb_embed.to(device)
    gt_xforms = gt_xforms.to(device)

    # exit(0)
    
    gt_color = [31, 119, 180, 255]
    pred_mesh_path = os.path.join(results_dir, 'pred_mesh.png')
    prex_pred_mesh_path = os.path.join(results_dir, 'prex_pred_mesh.png')
    obbs_path = os.path.join(results_dir, 'obbs.png')
    prex_obbs_path = os.path.join(results_dir, 'prex_obbs.png') # pre-xform
    lst_paths = [
        prex_obbs_path,
        prex_pred_mesh_path,
        obbs_path,
        pred_mesh_path
        ]

    # normal
    query_points = reconstruct.make_query_points(pt_sample_res)

    # explosion
    # limits=[(-0.525, 0.525)]*3
    # query_points = reconstruct.make_query_points(
    #     pt_sample_res, limits=limits)

    query_points = torch.from_numpy(query_points).to(device, torch.float32)
    query_points = query_points.unsqueeze(0)
    bs, num_points, _ = query_points.shape

    with torch.no_grad():
        if args.mask:
            # parts = [args.part]
            parts = args.parts
            parts = [int(x) for x in parts]
            if args.recon_one_part:
                # only reconstruct part specified by parts
                masked_indices = list(set(range(num_parts)) - set(parts))
                masked_indices = torch.Tensor(masked_indices).to(device, torch.long)
            if args.recon_the_rest:
                # don't reconstruct stuff in masked_indices
                masked_indices = torch.Tensor(parts).to(device, torch.long)
        else:
            masked_indices = torch.Tensor([]).to(device, torch.long)

        if not args.mask:
            print("reconstructing...")
        else:
            print(f"masking parts: {masked_indices.cpu().numpy().tolist()}")

        parts_mask = torch.zeros((batch_embed.shape[0], num_parts)).to(device, torch.float32)
        parts_mask[:, masked_indices] = 1
        parts_mask = torch.repeat_interleave(parts_mask, each_part_feat, dim=-1)
        
        points_mask = torch.zeros((1, num_parts, 1, 1)).to(device, torch.float32)
        points_mask[:, masked_indices] = 1

        occ_mask = torch.zeros((1, 1, num_parts)).to(device, torch.float32)
        occ_mask[:, :, masked_indices] = 1

        # learning xforms
        batch_vec = torch.arange(start=0, end=1).to(device)
        batch_vec = torch.repeat_interleave(batch_vec, num_union_nodes)
        batch_mask = torch.sum(batch_part_nodes, dim=1)
        learned_geom, learned_xforms = model.learn_geom_xform(batch_node_feat,
                                                    batch_adj,
                                                    batch_mask,
                                                    batch_vec)
        learned_relations = learned_xforms[:, connectivity[:, 0], connectivity[:, 1], :]
        learned_xforms = learned_xforms[:, [0, 1, 2, 3], [0, 1, 2, 3], :]

        batch_geom = torch.einsum('ijk, ikm -> ijm',
                                batch_part_nodes.to(torch.float32),
                                batch_node_feat)

        pairwise_xforms = learned_xforms[:, connectivity]
        learned_relations = pairwise_xforms[:, :, 1] - pairwise_xforms[:, :, 0]

        transformed_points = query_points.unsqueeze(1).expand(-1, num_parts, -1, -1) +\
            gt_xforms.unsqueeze(2)
        occs1 = model(transformed_points.masked_fill(points_mask==1,torch.tensor(0)),
                      batch_embed.masked_fill(parts_mask==1, torch.tensor(0)))
        occs1 = occs1.masked_fill(occ_mask==1,torch.tensor(float('-inf')))
        pred_values1, _ = torch.max(occs1, dim=-1, keepdim=True)

        transformed_points = query_points.unsqueeze(1).expand(-1, num_parts, -1, -1) +\
            learned_xforms.unsqueeze(2)
        occs2 = model(transformed_points.masked_fill(points_mask==1,torch.tensor(0)),
                      batch_embed.masked_fill(parts_mask==1, torch.tensor(0)))
        occs2 = occs2.masked_fill(occ_mask==1,torch.tensor(float('-inf')))
        pred_values2, _ = torch.max(occs2, dim=-1, keepdim=True)

    learned_xforms = learned_xforms[0].cpu().numpy()
    learned_geom = learned_geom[0].cpu().numpy()
    gt_xforms = gt_xforms[0].cpu().numpy()

    obbs_of_interest = [[]] * num_parts
    prex_obbs_of_interest = [[]] * num_parts

    for i in range(num_parts):
        ext = gt_exts[i].numpy()
        learned_xform = np.eye(4)
        learned_xform[:3, 3] = -learned_xforms[i]
        gt_xform = np.eye(4)
        gt_xform[:3, 3] = -gt_xforms[i]
        learned_ext = learned_geom[i]
        # ext_xform = (learned_ext, learned_xform)
        ext_xform = (ext, learned_xform)
        prex_ext_xform = (ext, gt_xform)
        obbs_of_interest[i] = [ext_xform]
        prex_obbs_of_interest[i] = [prex_ext_xform]

    if args.mask:
        mag = 1
    else:
        mag = 0.8

    sdf_grid = torch.reshape(
        pred_values1,
        (1, pt_sample_res, pt_sample_res, pt_sample_res))
    sdf_grid = torch.permute(sdf_grid, (0, 2, 1, 3))
    vertices, faces =\
        kaolin.ops.conversions.voxelgrids_to_trianglemeshes(sdf_grid)
    vertices = kaolin.ops.pointcloud.center_points(
        vertices[0].unsqueeze(0), normalize=True).squeeze(0)
    pred_vertices = vertices.cpu().numpy()
    pred_faces = faces[0].cpu().numpy()
    prex_pred_mesh = trimesh.Trimesh(pred_vertices, pred_faces)
    prex_pred_mesh.export(os.path.join(results_dir, 'prex_mesh_pred.obj'))
    visualize.save_mesh_vis(prex_pred_mesh, prex_pred_mesh_path,
                            mag=mag, white_bg=True)
    
    sdf_grid = torch.reshape(
        pred_values2,
        (1, pt_sample_res, pt_sample_res, pt_sample_res))
    sdf_grid = torch.permute(sdf_grid, (0, 2, 1, 3))
    vertices, faces =\
        kaolin.ops.conversions.voxelgrids_to_trianglemeshes(sdf_grid)
    vertices = kaolin.ops.pointcloud.center_points(
        vertices[0].unsqueeze(0), normalize=True).squeeze(0)
    pred_vertices = vertices.cpu().numpy()
    pred_faces = faces[0].cpu().numpy()
    pred_mesh = trimesh.Trimesh(pred_vertices, pred_faces)
    pred_mesh.export(os.path.join(results_dir, 'mesh_pred.obj'))
    visualize.save_mesh_vis(pred_mesh, pred_mesh_path,
                            mag=mag, white_bg=True)
    
    # exit(0)
    masked_indices = masked_indices.cpu().numpy()
    unmasked_indices = list(set(range(num_parts)) - set(masked_indices))

    import itertools
    obbs_of_interest = [obbs_of_interest[x] for x in unmasked_indices]
    obbs_of_interest = list(itertools.chain(*obbs_of_interest))
    visualize.save_obbs_vis(obbs_of_interest,
                            obbs_path, mag=mag, white_bg=True,
                            unmasked_indices=unmasked_indices)
    
    prex_obbs_of_interest = [prex_obbs_of_interest[x] for x in unmasked_indices]
    prex_obbs_of_interest = list(itertools.chain(*prex_obbs_of_interest))
    visualize.save_obbs_vis(prex_obbs_of_interest,
                            prex_obbs_path, mag=mag, white_bg=True,
                            unmasked_indices=unmasked_indices)
    
    if not args.mask:
        visualize.stitch_imges(
            os.path.join(results_dir,f'assembly_results.png'),
            image_paths=lst_paths,
            adj=100)
    else:
        # parts_str contains all the parts that are MASKED OUT
        parts_str = '-'.join([str(x) for x in masked_indices])
        visualize.stitch_imges(
            os.path.join(results_dir,f'assembly_results_mask_{parts_str}.png'),
            image_paths=lst_paths,
            adj=100)
        
    exit(0)


if args.inv and not args.sc and not args.asb_scaling:
    from utils import visualize

    white_bg = True
    it = args.it
    model_idx = args.test_idx
    anno_id = model_idx_to_anno_id[model_idx]
    model_id = misc.anno_id_to_model_id(partnet_index_path)[anno_id]
    print(f"anno id: {anno_id}, model id: {model_id}")

    checkpoint = torch.load(os.path.join(ckpt_dir, f'model_{it}.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    results_dir = os.path.join(results_dir, anno_id)
    misc.check_dir(results_dir)

    embedding_fp = os.path.join(results_dir, 'embedding.pth')
    if os.path.exists(embedding_fp):
        embeddings = torch.load(embedding_fp).to(device)
    else:
        embeddings = torch.nn.Embedding(1, num_parts*each_part_feat).to(device)
        torch.nn.init.normal_(
            embeddings.weight.data,
            0.0,
            1 / math.sqrt(num_parts*each_part_feat),
        )

    unique_part_names, name_to_ori_ids_and_objs,\
        orig_obbs, entire_mesh, name_to_obbs, _, _, _ =\
        preprocess_data_17.merge_partnet_after_merging(anno_id)

    with open(f'data/{cat_name}_part_name_to_new_id_17_{ds_start}_{ds_end}.json', 'r') as f:
        unique_name_to_new_id = json.load(f)

    part_obbs = []

    unique_names = list(unique_name_to_new_id.keys())
    model_part_names = list(name_to_obbs.keys())

    for i, un in enumerate(unique_names):
        if not un in model_part_names:
            part_obbs.append([])
            continue
        part_obbs.append([name_to_obbs[un]])

    gt_color = [31, 119, 180, 255]
    mesh_pred_path = os.path.join(results_dir, 'mesh_pred.png')
    mesh_gt_path = os.path.join(results_dir, 'mesh_gt.png')
    learned_obbs_path = os.path.join(results_dir, 'obbs_pred.png')
    obbs_path = os.path.join(results_dir, 'obbs_gt.png')
    lst_paths = [
        obbs_path,
        mesh_gt_path,
        learned_obbs_path,
        mesh_pred_path]

    batch_node_feat = torch.from_numpy(node_features[model_idx:model_idx+1, :, :3]).to(device, torch.float32)
    batch_adj = torch.from_numpy(adj[model_idx:model_idx+1]).to(device, torch.float32)
    batch_part_nodes = torch.from_numpy(part_nodes[model_idx:model_idx+1]).to(device, torch.float32)
    batch_xforms = torch.from_numpy(xforms[model_idx:model_idx+1, :, :3, 3]).to(device, torch.float32)
    batch_relations = torch.from_numpy(relations[model_idx:model_idx+1, :, :3, 3]).to(device, torch.float32)

    gt_points = torch.from_numpy(normalized_points[model_idx:model_idx+1]).to(device, torch.float32)
    gt_values = torch.from_numpy(values[model_idx:model_idx+1]).to(device, torch.float32)
    
    optimizer = torch.optim.Adam([{"params": embeddings.parameters(), "lr": lr}])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda x: max(0.0, 10**(-x*0.0002)))
    num_iterations = 3000

    if not os.path.exists(embedding_fp):
        print("optimizing for embedding")
        for i in tqdm(range(num_iterations)):
            batch_embed = embeddings(torch.tensor(0).to(device)).unsqueeze(0)
            batch_vec = torch.arange(start=0, end=1).to(device)
            batch_vec = torch.repeat_interleave(batch_vec, num_union_nodes)
            batch_mask = torch.sum(batch_part_nodes, dim=1)
            with torch.no_grad():
                learned_geom, learned_xforms = model.learn_geom_xform(batch_node_feat,
                                                            batch_adj,
                                                            batch_mask,
                                                            batch_vec)
                learned_relations = learned_xforms[:, connectivity[:, 0], connectivity[:, 1], :]
                learned_xforms = learned_xforms[:, [0, 1, 2, 3], [0, 1, 2, 3], :]
            # learned_xforms = batch_xforms

            batch_geom = torch.einsum('ijk, ikm -> ijm',
                                    batch_part_nodes.to(torch.float32),
                                    batch_node_feat)

            pairwise_xforms = learned_xforms[:, connectivity]
            # learned_relations = pairwise_xforms[:, :, 1] - pairwise_xforms[:, :, 0]

            transformed_points = gt_points.unsqueeze(1).expand(-1, num_parts, -1, -1) +\
                learned_xforms.unsqueeze(2)

            occs = model(transformed_points, batch_embed)
            pred_values, _ = torch.max(occs, dim=-1, keepdim=True)

            loss = loss_f(pred_values, gt_values)

            # loss += 10 * loss_xform + 10 * loss_relations

            # if (i+1) % 100 == 0:  # Print loss every 100 iterations
            #     print(f'Iteration {i+1}/{num_iterations}, Loss: {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        print(f'Loss: {loss.item()}')

    if not os.path.exists(embedding_fp):
        # save optimized embeddings
        # embedding_fp = os.path.join(results_dir, 'embedding.pth')
        torch.save(embeddings, embedding_fp)

    query_points = reconstruct.make_query_points(pt_sample_res)
    query_points = torch.from_numpy(query_points).to(device, torch.float32)
    query_points = query_points.unsqueeze(0)
    bs, num_points, _ = query_points.shape

    with torch.no_grad():
        batch_embed = embeddings(torch.tensor(0).to(device)).unsqueeze(0)
        if args.mask:
            # parts = [args.part]
            parts = args.parts
            parts = [int(x) for x in parts]
            if args.recon_one_part:
                # only reconstruct part specified by parts
                masked_indices = list(set(range(num_parts)) - set(parts))
                masked_indices = torch.Tensor(masked_indices).to(device, torch.long)
            if args.recon_the_rest:
                # don't reconstruct stuff in masked_indices
                masked_indices = torch.Tensor(parts).to(device, torch.long)
        else:
            masked_indices = torch.Tensor([]).to(device, torch.long)

        if not args.mask:
            print("reconstructing...")
        else:
            print(f"masking parts: {masked_indices.cpu().numpy().tolist()}")

        parts_mask = torch.zeros((batch_embed.shape[0], num_parts)).to(device, torch.float32)
        parts_mask[:, masked_indices] = 1
        parts_mask = torch.repeat_interleave(parts_mask, each_part_feat, dim=-1)
        
        points_mask = torch.zeros((1, num_parts, 1, 1)).to(device, torch.float32)
        points_mask[:, masked_indices] = 1

        occ_mask = torch.zeros((1, 1, num_parts)).to(device, torch.float32)
        occ_mask[:, :, masked_indices] = 1

        # learning xforms
        batch_vec = torch.arange(start=0, end=1).to(device)
        batch_vec = torch.repeat_interleave(batch_vec, num_union_nodes)
        batch_mask = torch.sum(batch_part_nodes, dim=1)
        learned_geom, learned_xforms = model.learn_geom_xform(batch_node_feat,
                                                    batch_adj,
                                                    batch_mask,
                                                    batch_vec)
        learned_relations = learned_xforms[:, connectivity[:, 0], connectivity[:, 1], :]
        learned_xforms = learned_xforms[:, [0, 1, 2, 3], [0, 1, 2, 3], :]

        batch_geom = torch.einsum('ijk, ikm -> ijm',
                                batch_part_nodes.to(torch.float32),
                                batch_node_feat)

        pairwise_xforms = learned_xforms[:, connectivity]
        learned_relations = pairwise_xforms[:, :, 1] - pairwise_xforms[:, :, 0]

        transformed_points = query_points.unsqueeze(1).expand(-1, num_parts, -1, -1) +\
            learned_xforms.unsqueeze(2)

        occs1 = model(transformed_points.masked_fill(points_mask==1,torch.tensor(0)),
                      batch_embed.masked_fill(parts_mask==1, torch.tensor(0)))
        occs1 = occs1.masked_fill(occ_mask==1,torch.tensor(float('-inf')))
        pred_values, _ = torch.max(occs1, dim=-1, keepdim=True)


    learned_xforms = learned_xforms[0].cpu().numpy()
    learned_geom = learned_geom[0].cpu().numpy()
    learned_obbs_of_interest = [[]] * num_parts
    for i in range(num_parts):
        ext = extents[model_idx, i]
        learned_xform = np.eye(4)
        learned_xform[:3, 3] = -learned_xforms[i]
        learned_ext = learned_geom[i]
        # ext_xform = (ext, learned_xform)
        ext_xform = (learned_ext, learned_xform)
        learned_obbs_of_interest[i] = [ext_xform]

    if args.mask:
        mag = 1
    else:
        mag = 0.8

    sdf_grid = torch.reshape(
        pred_values,
        (1, pt_sample_res, pt_sample_res, pt_sample_res))
    sdf_grid = torch.permute(sdf_grid, (0, 2, 1, 3))
    vertices, faces =\
        kaolin.ops.conversions.voxelgrids_to_trianglemeshes(sdf_grid)
    vertices = kaolin.ops.pointcloud.center_points(
        vertices[0].unsqueeze(0), normalize=True).squeeze(0)
    pred_vertices = vertices.cpu().numpy()
    pred_faces = faces[0].cpu().numpy()
    pred_mesh = trimesh.Trimesh(pred_vertices, pred_faces)
    pred_mesh.export(os.path.join(results_dir, 'mesh_pred.obj'))
    visualize.save_mesh_vis(pred_mesh, mesh_pred_path,
                            mag=mag, white_bg=white_bg)
    
    unmasked_indices = list(set(range(num_parts)) - set(masked_indices))

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

        masked_indices = masked_indices.cpu().numpy().tolist()
        unmasked_indices = list(set(range(num_parts)) - set(masked_indices))

        if args.recon_one_part:
            # only show stuff in masked_indices
            concat_mesh = trimesh.util.concatenate(
                [lst_of_part_meshes[x] for x in parts])
        if args.recon_the_rest:
            # don't show stuff in masked_indices
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
        pred_mesh, 10000, seed=319)
    pred_samples = torch.from_numpy(pred_samples).to(device)
    chamfer = kaolin.metrics.pointcloud.chamfer_distance(
        pred_samples.unsqueeze(0), gt_samples.unsqueeze(0)).mean()
    print("[EVAL] Chamfer distance: ", chamfer.cpu().numpy())

    import itertools
    obbs_of_interest = [part_obbs[x] for x in unmasked_indices]
    obbs_of_interest = list(itertools.chain(*obbs_of_interest))
    visualize.save_obbs_vis(obbs_of_interest,
                            obbs_path, mag=mag, white_bg=white_bg,
                            unmasked_indices=unmasked_indices)
    
    learned_obbs_of_interest = [learned_obbs_of_interest[x] for x in unmasked_indices]
    learned_obbs_of_interest = list(itertools.chain(*learned_obbs_of_interest))
    visualize.save_obbs_vis(learned_obbs_of_interest,
                            learned_obbs_path, mag=mag, white_bg=white_bg,
                            unmasked_indices=unmasked_indices)
    
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


if args.samp:
    from utils import visualize

    white_bg = True
    it = args.it
    model_idx = args.test_idx
    sample_idx = args.sample_idx
    anno_id = model_idx_to_anno_id[model_idx]
    model_id = misc.anno_id_to_model_id(partnet_index_path)[anno_id]
    print(f"anno id: {anno_id}, model id: {model_id}")

    checkpoint = torch.load(os.path.join(ckpt_dir, f'model_{it}.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    embeddings = torch.nn.Embedding(num_shapes, num_parts*each_part_feat)
    embeddings.load_state_dict(checkpoint['embeddings_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    results_dir = os.path.join(results_dir, f'{anno_id}-samples', f'{anno_id}-sample-{sample_idx}')
    misc.check_dir(results_dir)
    print(results_dir)

    def sample_pca_space(xformed_pca, num_samples=10, scale=1.0):
        mean = np.mean(xformed_pca, axis=0)
        std_dev = np.std(xformed_pca, axis=0)
        samples = np.random.normal(mean, std_dev * scale, size=(num_samples, xformed_pca.shape[1]))
        return samples

    # # sample by PCA, entire vector
    # NOTE: generated shapes are cohesive!
    np.random.seed(319)
    num_samples = 20
    scale = 1.0  # Adjust scale to control the diversity of generated shapes
    pca = PCA(n_components=4)  # Number of components should be <= embedding_dim
    pca.fit(embeddings.weight.data.numpy()[:500])
    embedddings_pca = pca.transform(embeddings.weight.data.numpy()[:500])
    sampled_pca_embeddings = sample_pca_space(embedddings_pca, num_samples, scale)
    sampled_lat_embeddings = pca.inverse_transform(sampled_pca_embeddings)
    sampled_lat_embeddings = torch.from_numpy(sampled_lat_embeddings)
    batch_embed = sampled_lat_embeddings[sample_idx].to(device, torch.float32).unsqueeze(0)

    # # sampe by PCA, by part
    # # NOTE: generated shapes aren't cohesive! (because there's no global notion)
    # # does this mean if i don't run loss on entire shapes (instead only on local shapes)
    # # during training, i can't sample cohesive shapes??
    # np.random.seed(319)
    # num_samples = 20
    # scale = 1.0
    # sampled_lat_embeddings = torch.zeros(num_samples, num_parts*each_part_feat)
    # for i in range(num_parts):
    #     pca = PCA(n_components=8)
    #     chunk = embeddings.weight.data.numpy()[:500, i*each_part_feat:(i+1)*each_part_feat]
    #     pca.fit(chunk)
    #     embedddings_pca = pca.transform(chunk)
    #     part_pca_embeddings = sample_pca_space(embedddings_pca, num_samples, scale)
    #     part_lat_embeddings = pca.inverse_transform(part_pca_embeddings)
    #     sampled_lat_embeddings[:, i*each_part_feat:(i+1)*each_part_feat] =\
    #         torch.from_numpy(part_lat_embeddings)
    # sampled_lat_embeddings = sampled_lat_embeddings.to(device, torch.float32)
    # batch_embed = sampled_lat_embeddings[sample_idx].unsqueeze(0)

    unique_part_names, name_to_ori_ids_and_objs,\
        orig_obbs, entire_mesh, name_to_obbs, _, _, _ =\
        preprocess_data_17.merge_partnet_after_merging(anno_id)

    with open(f'data/{cat_name}_part_name_to_new_id_17_{ds_start}_{ds_end}.json', 'r') as f:
        unique_name_to_new_id = json.load(f)

    part_obbs = []

    unique_names = list(unique_name_to_new_id.keys())
    model_part_names = list(name_to_obbs.keys())

    for i, un in enumerate(unique_names):
        if not un in model_part_names:
            part_obbs.append([])
            continue
        part_obbs.append([name_to_obbs[un]])

    gt_color = [31, 119, 180, 255]
    mesh_pred_path = os.path.join(results_dir, 'mesh_pred.png')
    mesh_gt_path = os.path.join(results_dir, 'mesh_gt.png')
    learned_obbs_path = os.path.join(results_dir, 'obbs_pred.png')
    obbs_path = os.path.join(results_dir, 'obbs_gt.png')
    lst_paths = [
        # obbs_path,
        # mesh_gt_path,
        learned_obbs_path,
        mesh_pred_path]

    query_points = reconstruct.make_query_points(pt_sample_res)
    query_points = torch.from_numpy(query_points).to(device, torch.float32)
    query_points = query_points.unsqueeze(0)
    bs, num_points, _ = query_points.shape

    batch_node_feat = torch.from_numpy(node_features[model_idx:model_idx+1, :, :3]).to(device, torch.float32)
    batch_adj = torch.from_numpy(adj[model_idx:model_idx+1]).to(device, torch.float32)
    batch_part_nodes = torch.from_numpy(part_nodes[model_idx:model_idx+1]).to(device, torch.float32)

    with torch.no_grad():
        if args.mask:
            # parts = [args.part]
            parts = args.parts
            parts = [int(x) for x in parts]
            if args.recon_one_part:
                # only reconstruct part specified by parts
                masked_indices = list(set(range(num_parts)) - set(parts))
                masked_indices = torch.Tensor(masked_indices).to(device, torch.long)
            if args.recon_the_rest:
                # don't reconstruct stuff in masked_indices
                masked_indices = torch.Tensor(parts).to(device, torch.long)
        else:
            masked_indices = torch.Tensor([]).to(device, torch.long)

        if not args.mask:
            print("reconstructing...")
        else:
            print(f"masking parts: {masked_indices.cpu().numpy().tolist()}")

        parts_mask = torch.zeros((batch_embed.shape[0], num_parts)).to(device, torch.float32)
        parts_mask[:, masked_indices] = 1
        parts_mask = torch.repeat_interleave(parts_mask, each_part_feat, dim=-1)
        
        points_mask = torch.zeros((1, num_parts, 1, 1)).to(device, torch.float32)
        points_mask[:, masked_indices] = 1

        occ_mask = torch.zeros((1, 1, num_parts)).to(device, torch.float32)
        occ_mask[:, :, masked_indices] = 1

        # learning xforms
        batch_vec = torch.arange(start=0, end=1).to(device)
        batch_vec = torch.repeat_interleave(batch_vec, num_union_nodes)
        batch_mask = torch.sum(batch_part_nodes, dim=1)
        learned_geom, learned_xforms = model.learn_geom_xform(batch_node_feat,
                                                    batch_adj,
                                                    batch_mask,
                                                    batch_vec)
        learned_relations = learned_xforms[:, connectivity[:, 0], connectivity[:, 1], :]
        learned_xforms = learned_xforms[:, [0, 1, 2, 3], [0, 1, 2, 3], :]

        batch_geom = torch.einsum('ijk, ikm -> ijm',
                                batch_part_nodes.to(torch.float32),
                                batch_node_feat)

        pairwise_xforms = learned_xforms[:, connectivity]
        learned_relations = pairwise_xforms[:, :, 1] - pairwise_xforms[:, :, 0]

        transformed_points = query_points.unsqueeze(1).expand(-1, num_parts, -1, -1) +\
            learned_xforms.unsqueeze(2)

        occs1 = model(transformed_points.masked_fill(points_mask==1,torch.tensor(0)),
                      batch_embed.masked_fill(parts_mask==1, torch.tensor(0)))
        occs1 = occs1.masked_fill(occ_mask==1,torch.tensor(float('-inf')))
        pred_values, _ = torch.max(occs1, dim=-1, keepdim=True)

    learned_xforms = learned_xforms[0].cpu().numpy()
    learned_obbs_of_interest = [[]] * num_parts
    for i in range(num_parts):
        ext = extents[model_idx, i]
        learned_xform = np.eye(4)
        learned_xform[:3, 3] = -learned_xforms[i]
        ext_xform = (ext, learned_xform)
        learned_obbs_of_interest[i] = [ext_xform]

    if args.mask:
        mag = 1
    else:
        mag = 0.8

    sdf_grid = torch.reshape(
        pred_values,
        (1, pt_sample_res, pt_sample_res, pt_sample_res))
    sdf_grid = torch.permute(sdf_grid, (0, 2, 1, 3))
    vertices, faces =\
        kaolin.ops.conversions.voxelgrids_to_trianglemeshes(sdf_grid)
    vertices = kaolin.ops.pointcloud.center_points(
        vertices[0].unsqueeze(0), normalize=True).squeeze(0)
    pred_vertices = vertices.cpu().numpy()
    pred_faces = faces[0].cpu().numpy()
    pred_mesh = trimesh.Trimesh(pred_vertices, pred_faces)
    pred_mesh.export(os.path.join(results_dir, 'mesh_pred.obj'))
    visualize.save_mesh_vis(pred_mesh, mesh_pred_path,
                            mag=mag, white_bg=white_bg)
    
    unmasked_indices = list(set(range(num_parts)) - set(masked_indices))

    import itertools    
    learned_obbs_of_interest = [learned_obbs_of_interest[x] for x in unmasked_indices]
    learned_obbs_of_interest = list(itertools.chain(*learned_obbs_of_interest))
    visualize.save_obbs_vis(learned_obbs_of_interest,
                            learned_obbs_path, mag=mag, white_bg=white_bg,
                            unmasked_indices=unmasked_indices)
    
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


if args.shape_complete:
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import NearestNeighbors
    from utils import visualize

    white_bg = True
    it = args.it
    model_idx = args.test_idx
    sample_idx = args.sample_idx
    fixed_parts = args.fixed_indices
    fixed_parts = [int(x) for x in fixed_parts]
    unfixed_parts = list(set(range(num_parts)) - set(fixed_parts))
    anno_id = model_idx_to_anno_id[model_idx]
    model_id = misc.anno_id_to_model_id(partnet_index_path)[anno_id]
    print(f"anno id: {anno_id}, model id: {model_id}")

    checkpoint = torch.load(os.path.join(ckpt_dir, f'model_{it}.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # if not args.inv:
    #     embeddings = torch.nn.Embedding(num_shapes, num_parts*each_part_feat)
    #     embeddings.load_state_dict(checkpoint['embeddings_state_dict'])
    # else:
    #     embeddings = torch.zeros(num_shapes, num_parts*each_part_feat)
    #     all_paths = [os.path.join(old_results_dir, str(x), 'embedding.pth') for x in anno_ids]
    #     for p in all_paths:
    #         assert os.path.exists(p), "you haven't run shape inversion for all unseen models"
    #     all_embeddings = [torch.load(p) for p in all_paths]
    #     for i in range(len(all_embeddings)):
    #         embeddings[i] = all_embeddings[i].weight.data[0]

    embeddings = torch.nn.Embedding(num_shapes, num_parts*each_part_feat)
    embeddings.load_state_dict(checkpoint['embeddings_state_dict'])

    old_results_dir = results_dir
    parts_str = '-'.join([str(x) for x in fixed_parts])
    results_dir = os.path.join(results_dir, f'{anno_id}-completion-{parts_str}', f'{anno_id}-sample-{sample_idx}')
    misc.check_dir(results_dir)
    print(results_dir)

    dim_fixed = len(fixed_parts) * each_part_feat
    dim_unfixed = num_parts * each_part_feat - dim_fixed

    fixed_embed = np.zeros((500, dim_fixed), np.float32)
    unfixed_embed = np.zeros((500, dim_unfixed), np.float32)

    for i, idx in enumerate(fixed_parts):
        fixed_embed[:500, i*each_part_feat:(i+1)*each_part_feat] =\
            embeddings.weight.data.numpy()[:500, idx*each_part_feat:(idx+1)*each_part_feat]
    
    for i, idx in enumerate(unfixed_parts):
        unfixed_embed[:500, i*each_part_feat:(i+1)*each_part_feat] =\
            embeddings.weight.data.numpy()[:500, idx*each_part_feat:(idx+1)*each_part_feat]

    nn_model = NearestNeighbors(n_neighbors=50)
    nn_model.fit(fixed_embed)

    def sample_conditional(fixed_vector, nn_model: NearestNeighbors, unfixed_part, num_samples=1):
        _, indices = nn_model.kneighbors(fixed_vector)
        nn_unfixed_embed = unfixed_part[indices[0]]

        # gmm = GaussianMixture(n_components=5, covariance_type='full')
        # gmm.fit(nn_unfixed_embed)

        # sampled_variance_parts = gmm.sample(num_samples)[0]
        # return sampled_variance_parts

        pca = PCA(n_components=4)  # Number of components should be <= embedding_dim
        pca.fit(nn_unfixed_embed)
        embedddings_pca = pca.transform(nn_unfixed_embed)
        sampled_pca_embeddings = sample_pca_space(embedddings_pca, num_samples)
        sampled_lat_embeddings = pca.inverse_transform(sampled_pca_embeddings)
        sampled_lat_embeddings = torch.from_numpy(sampled_lat_embeddings)
        return sampled_lat_embeddings
        
        # batch_embed = sampled_lat_embeddings[sample_idx].to(device, torch.float32).unsqueeze(0)
    
    def sample_pca_space(xformed_pca, num_samples=10, scale=1.0):
        mean = np.mean(xformed_pca, axis=0)
        std_dev = np.std(xformed_pca, axis=0)
        samples = np.random.normal(mean, std_dev * scale, size=(num_samples, xformed_pca.shape[1]))
        return samples

    # # sample by PCA, entire vector
    # NOTE: generated shapes are cohesive!
    np.random.seed(319)
    num_samples = 10

    if not args.inv:
        test_embed = embeddings.weight.data.numpy()[model_idx]
    else:
        embed_p = os.path.join(old_results_dir, str(anno_id), 'embedding.pth')
        assert os.path.exists(embed_p), "you haven't run shape inversion for all unseen models"
        test_embed = torch.load(embed_p).weight.data.cpu().numpy()[0]

    fixed_test_embed = np.concatenate([test_embed[i*each_part_feat:(i+1)*each_part_feat] for i in fixed_parts])[None, :]
    sampled_unfixed_parts = sample_conditional(fixed_test_embed, nn_model, unfixed_embed, num_samples)
    complete_embed = np.zeros((num_samples, num_parts*each_part_feat), np.float32)
    for i, idx in enumerate(fixed_parts):
        complete_embed[:, idx*each_part_feat:(idx+1)*each_part_feat] =\
            fixed_test_embed[:, i*each_part_feat:(i+1)*each_part_feat]
    for i, idx in enumerate(unfixed_parts):
        complete_embed[:, idx*each_part_feat:(idx+1)*each_part_feat] =\
            sampled_unfixed_parts[:, i*each_part_feat:(i+1)*each_part_feat]
    batch_embed = torch.from_numpy(complete_embed[sample_idx]).unsqueeze(0).to(device, torch.float32)

    unique_part_names, name_to_ori_ids_and_objs,\
        orig_obbs, entire_mesh, name_to_obbs, _, _, _ =\
        preprocess_data_17.merge_partnet_after_merging(anno_id)

    with open(f'data/{cat_name}_part_name_to_new_id_17_{ds_start}_{ds_end}.json', 'r') as f:
        unique_name_to_new_id = json.load(f)

    part_obbs = []

    unique_names = list(unique_name_to_new_id.keys())
    model_part_names = list(name_to_obbs.keys())

    for i, un in enumerate(unique_names):
        if not un in model_part_names:
            part_obbs.append([])
            continue
        part_obbs.append([name_to_obbs[un]])

    gt_color = [31, 119, 180, 255]
    mesh_pred_path = os.path.join(results_dir, 'mesh_pred.png')
    mesh_gt_path = os.path.join(results_dir, 'mesh_gt.png')
    learned_obbs_path = os.path.join(results_dir, 'obbs_pred.png')
    obbs_path = os.path.join(results_dir, 'obbs_gt.png')
    lst_paths = [
        # obbs_path,
        # mesh_gt_path,
        learned_obbs_path,
        mesh_pred_path]

    query_points = reconstruct.make_query_points(pt_sample_res)
    query_points = torch.from_numpy(query_points).to(device, torch.float32)
    query_points = query_points.unsqueeze(0)
    bs, num_points, _ = query_points.shape

    batch_node_feat = torch.from_numpy(node_features[model_idx:model_idx+1, :, :3]).to(device, torch.float32)
    batch_adj = torch.from_numpy(adj[model_idx:model_idx+1]).to(device, torch.float32)
    batch_part_nodes = torch.from_numpy(part_nodes[model_idx:model_idx+1]).to(device, torch.float32)

    with torch.no_grad():
        if args.mask:
            # parts = [args.part]
            parts = args.parts
            parts = [int(x) for x in parts]
            if args.recon_one_part:
                # only reconstruct part specified by parts
                masked_indices = list(set(range(num_parts)) - set(parts))
                masked_indices = torch.Tensor(masked_indices).to(device, torch.long)
            if args.recon_the_rest:
                # don't reconstruct stuff in masked_indices
                masked_indices = torch.Tensor(parts).to(device, torch.long)
        else:
            masked_indices = torch.Tensor([]).to(device, torch.long)

        if not args.mask:
            print("reconstructing...")
        else:
            print(f"masking parts: {masked_indices.cpu().numpy().tolist()}")

        parts_mask = torch.zeros((batch_embed.shape[0], num_parts)).to(device, torch.float32)
        parts_mask[:, masked_indices] = 1
        parts_mask = torch.repeat_interleave(parts_mask, each_part_feat, dim=-1)
        
        points_mask = torch.zeros((1, num_parts, 1, 1)).to(device, torch.float32)
        points_mask[:, masked_indices] = 1

        occ_mask = torch.zeros((1, 1, num_parts)).to(device, torch.float32)
        occ_mask[:, :, masked_indices] = 1

        # learning xforms
        batch_vec = torch.arange(start=0, end=1).to(device)
        batch_vec = torch.repeat_interleave(batch_vec, num_union_nodes)
        batch_mask = torch.sum(batch_part_nodes, dim=1)
        learned_geom, learned_xforms = model.learn_geom_xform(batch_node_feat,
                                                    batch_adj,
                                                    batch_mask,
                                                    batch_vec)
        learned_relations = learned_xforms[:, connectivity[:, 0], connectivity[:, 1], :]
        learned_xforms = learned_xforms[:, [0, 1, 2, 3], [0, 1, 2, 3], :]

        batch_geom = torch.einsum('ijk, ikm -> ijm',
                                batch_part_nodes.to(torch.float32),
                                batch_node_feat)

        pairwise_xforms = learned_xforms[:, connectivity]
        learned_relations = pairwise_xforms[:, :, 1] - pairwise_xforms[:, :, 0]

        transformed_points = query_points.unsqueeze(1).expand(-1, num_parts, -1, -1) +\
            learned_xforms.unsqueeze(2)

        occs1 = model(transformed_points.masked_fill(points_mask==1,torch.tensor(0)),
                      batch_embed.masked_fill(parts_mask==1, torch.tensor(0)))
        occs1 = occs1.masked_fill(occ_mask==1,torch.tensor(float('-inf')))
        pred_values, _ = torch.max(occs1, dim=-1, keepdim=True)

    learned_xforms = learned_xforms[0].cpu().numpy()
    learned_obbs_of_interest = [[]] * num_parts
    for i in range(num_parts):
        ext = extents[model_idx, i]
        learned_xform = np.eye(4)
        learned_xform[:3, 3] = -learned_xforms[i]
        ext_xform = (ext, learned_xform)
        learned_obbs_of_interest[i] = [ext_xform]

    if args.mask:
        mag = 0.8
    else:
        mag = 0.8

    sdf_grid = torch.reshape(
        pred_values,
        (1, pt_sample_res, pt_sample_res, pt_sample_res))
    sdf_grid = torch.permute(sdf_grid, (0, 2, 1, 3))
    vertices, faces =\
        kaolin.ops.conversions.voxelgrids_to_trianglemeshes(sdf_grid)
    vertices = kaolin.ops.pointcloud.center_points(
        vertices[0].unsqueeze(0), normalize=True).squeeze(0)
    pred_vertices = vertices.cpu().numpy()
    pred_faces = faces[0].cpu().numpy()
    pred_mesh = trimesh.Trimesh(pred_vertices, pred_faces)
    pred_mesh.export(os.path.join(results_dir, 'mesh_pred.obj'))
    visualize.save_mesh_vis(pred_mesh, mesh_pred_path,
                            mag=mag, white_bg=white_bg)
    
    unmasked_indices = list(set(range(num_parts)) - set(masked_indices))

    import itertools    
    learned_obbs_of_interest = [learned_obbs_of_interest[x] for x in unmasked_indices]
    learned_obbs_of_interest = list(itertools.chain(*learned_obbs_of_interest))
    visualize.save_obbs_vis(learned_obbs_of_interest,
                            learned_obbs_path, mag=mag, white_bg=white_bg,
                            unmasked_indices=unmasked_indices)
    
    if not args.mask:
        visualize.stitch_imges(
            os.path.join(results_dir,f'{anno_id}_results.png'),
            image_paths=lst_paths,
            adj=100)
    else:
        # parts_str contains all the parts that are MASKED OUT
        parts_str = '-'.join([str(x) for x in masked_indices.cpu().numpy().tolist()])
        visualize.stitch_imges(
            os.path.join(results_dir,f'{anno_id}_results_mask_{parts_str}.png'),
            image_paths=lst_paths,
            adj=100)

    exit(0)


if args.asb_scaling:
    from utils import visualize

    # assembly shape from coarse parts
    it = args.it

    # model_indices = [500, 501, 506, 505]
    # part_indices = [1, 0, 3, 2]
    # model_indices = [500, 501, 502, 505]
    # part_indices = [3, 0, 2, 1]
    # part_indices = [3, 0, 1, 2]
    # model_indices = [343, 185, 370, 258]
    # part_indices = [0, 2, 1, 3]
    # part_indices = [0, 2, 3, 1]
    # model_indices = [255, 354, 28, 175]
    # part_indices = [0, 2, 3, 1]
    # part_indices = [3, 1, 2, 0]

    # anno_ids = [model_idx_to_anno_id[x] for x in model_indices]
    
    anno_ids = ['47914', '44979', '2243', '44164']
    # anno_ids = ['37900', '39901', '37121', '2243']
    # anno_ids = ['36685', '38091', '38567', '38816']
    # anno_ids = ['38567', '36685', '2673', '38816']
    part_indices = [0, 1, 2, 3]

    anno_id_to_model_idx = {v: k for k, v in model_idx_to_anno_id.items()}
    model_indices = [anno_id_to_model_idx[x] for x in anno_ids]
    print(model_indices)
    print(anno_ids)

    asb_str = '-'.join([str(x) for x in anno_ids+part_indices])
    old_results_dir = results_dir
    results_dir = os.path.join(results_dir, f'0assembly_{asb_str}')
    misc.check_dir(results_dir)
    print("results dir: ", results_dir)

    with open(os.path.join(results_dir, 'model_indices.txt'), 'w') as f:
        model_indices_str = '-'.join([str(x) for x in model_indices])
        part_indices_str = '-'.join([str(x) for x in part_indices])
        # f.write(anno_ids_str)
        f.writelines([model_indices_str, '\n', part_indices_str])

    checkpoint = torch.load(os.path.join(ckpt_dir, f'model_{it}.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    if not args.inv:
        embeddings = torch.nn.Embedding(num_shapes, num_parts*each_part_feat)
        embeddings.load_state_dict(checkpoint['embeddings_state_dict'])
    else:
        embeddings = torch.zeros(num_shapes, num_parts*each_part_feat)
        all_paths = [os.path.join(old_results_dir, str(x), 'embedding.pth') for x in anno_ids]
        for p in all_paths:
            assert os.path.exists(p), "you haven't run shape inversion for all unseen models"
        all_embeddings = [torch.load(p) for p in all_paths]
        for i in range(len(all_embeddings)):
            embeddings[i] = all_embeddings[i].weight.data[0]

    all_part_num_obbs = []
    # obbs_of_interest = [[]] * num_parts
    # node features just have the extents
    asb_node_feat = torch.zeros((num_union_nodes, 3)).to(torch.float32)
    asb_part_nodes = torch.zeros((num_parts, num_union_nodes)).to(torch.long)
    asb_adj = torch.zeros((num_union_nodes, num_union_nodes)).to(torch.long)
    asb_embed = torch.zeros(1, num_parts*each_part_feat).to(torch.float32)
    gt_xforms = torch.zeros((1, num_parts, 3)).to(torch.float32)
    gt_exts = torch.zeros((num_parts, 3)).to(torch.float32)
    for i, idx in enumerate(model_indices):
        anno_id = model_idx_to_anno_id[idx]
        part_idx = part_indices[i]
        shape_node_feat = torch.from_numpy(node_features[idx, :, :3])    # 7, 3
        shape_part_nodes = torch.from_numpy(part_nodes[idx])  # 4, 7
        shape_adj = torch.from_numpy(adj[idx])
        shape_xform = torch.from_numpy(xforms[idx, part_idx, :3, 3])
        # shape_part_mask = shape_part_nodes[part_idx].unsqueeze(1)    # 7, 1
        # asb_node_features += shape_node_feat * shape_part_mask
        union_node_indices =\
            torch.argwhere(shape_part_nodes[part_idx] == 1).flatten()
        shape_ext = shape_node_feat[union_node_indices]
        asb_node_feat[union_node_indices] = shape_ext
        
        asb_part_nodes[part_idx] = shape_part_nodes[part_idx]
        
        for uni in union_node_indices:
            build_adj_matrix(uni, asb_adj, shape_adj)

        if not args.inv:
            asb_embed[:, part_idx*each_part_feat:(part_idx+1)*each_part_feat] =\
                embeddings(torch.tensor(idx))[part_idx*each_part_feat:(part_idx+1)*each_part_feat]
        else:
            asb_embed[:, part_idx*each_part_feat:(part_idx+1)*each_part_feat] =\
                embeddings[i, part_idx*each_part_feat:(part_idx+1)*each_part_feat]

        gt_xforms[:, part_idx] = shape_xform
        gt_exts[part_idx] = shape_ext

    # build a tree that is a subset of the union tree (dense graph)
    # using the bounding boxes
    node_names = np.load(f'data/{cat_name}_union_node_names_17_{ds_start}_{ds_end}.npy')
    recon_root = tree.recon_tree(asb_adj.numpy(), node_names)
    UniqueDotExporter(recon_root,
                      indent=0,
                      nodenamefunc=lambda node: node.name,
                      nodeattrfunc=lambda node: "shape=box",).to_picture(
                          os.path.join(results_dir, 'asb_tree.png'))

    batch_node_feat = asb_node_feat.unsqueeze(0).to(device)
    batch_part_nodes = asb_part_nodes.unsqueeze(0).to(device)
    batch_adj = asb_adj.unsqueeze(0).to(device)
    batch_embed = asb_embed.to(device)
    gt_xforms = gt_xforms.to(device)

    # exit(0)
    
    gt_color = [31, 119, 180, 255]
    pred_mesh_path = os.path.join(results_dir, 'pred_mesh.png')
    prex_pred_mesh_path = os.path.join(results_dir, 'prex_pred_mesh.png')
    obbs_path = os.path.join(results_dir, 'obbs.png')
    prex_obbs_path = os.path.join(results_dir, 'prex_obbs.png') # pre-xform
    lst_paths = [
        prex_obbs_path,
        prex_pred_mesh_path,
        obbs_path,
        pred_mesh_path
        ]

    query_points = reconstruct.make_query_points(pt_sample_res)
    query_points = torch.from_numpy(query_points).to(device, torch.float32)
    query_points = query_points.unsqueeze(0)
    bs, num_points, _ = query_points.shape

    with torch.no_grad():
        if args.mask:
            # parts = [args.part]
            parts = args.parts
            parts = [int(x) for x in parts]
            if args.recon_one_part:
                # only reconstruct part specified by parts
                masked_indices = list(set(range(num_parts)) - set(parts))
                masked_indices = torch.Tensor(masked_indices).to(device, torch.long)
            if args.recon_the_rest:
                # don't reconstruct stuff in masked_indices
                masked_indices = torch.Tensor(parts).to(device, torch.long)
        else:
            masked_indices = torch.Tensor([]).to(device, torch.long)

        if not args.mask:
            print("reconstructing...")
        else:
            print(f"masking parts: {masked_indices.cpu().numpy().tolist()}")

        parts_mask = torch.zeros((batch_embed.shape[0], num_parts)).to(device, torch.float32)
        parts_mask[:, masked_indices] = 1
        parts_mask = torch.repeat_interleave(parts_mask, each_part_feat, dim=-1)
        
        points_mask = torch.zeros((1, num_parts, 1, 1)).to(device, torch.float32)
        points_mask[:, masked_indices] = 1

        occ_mask = torch.zeros((1, 1, num_parts)).to(device, torch.float32)
        occ_mask[:, :, masked_indices] = 1

        # learning xforms
        batch_vec = torch.arange(start=0, end=1).to(device)
        batch_vec = torch.repeat_interleave(batch_vec, num_union_nodes)
        batch_mask = torch.sum(batch_part_nodes, dim=1)
        learned_geom, learned_xforms = model.learn_geom_xform(batch_node_feat,
                                                    batch_adj,
                                                    batch_mask,
                                                    batch_vec)
        learned_relations = learned_xforms[:, connectivity[:, 0], connectivity[:, 1], :]
        learned_xforms = learned_xforms[:, [0, 1, 2, 3], [0, 1, 2, 3], :]

        batch_geom = torch.einsum('ijk, ikm -> ijm',
                                batch_part_nodes.to(torch.float32),
                                batch_node_feat)

        pairwise_xforms = learned_xforms[:, connectivity]
        learned_relations = pairwise_xforms[:, :, 1] - pairwise_xforms[:, :, 0]

        transformed_points = query_points.unsqueeze(1).expand(-1, num_parts, -1, -1) +\
            gt_xforms.unsqueeze(2)
        occs1 = model(transformed_points.masked_fill(points_mask==1,torch.tensor(0)),
                      batch_embed.masked_fill(parts_mask==1, torch.tensor(0)))
        occs1 = occs1.masked_fill(occ_mask==1,torch.tensor(float('-inf')))
        pred_values1, _ = torch.max(occs1, dim=-1, keepdim=True)

    # learned_xforms = learned_xforms[0].cpu().numpy()
    learned_geom = learned_geom[0].cpu().numpy()
    gt_xforms = gt_xforms[0].cpu().numpy()

    obbs_of_interest = [[]] * num_parts
    prex_obbs_of_interest = [[]] * num_parts

    scales = learned_geom / gt_exts.numpy()
    scales = np.clip(scales, 0.5, 1.5)

    # print(learned_geom)
    # print(gt_exts)
    # print(scales)
    # exit(0)

    for i in range(num_parts):
        ext = gt_exts[i].numpy()
        learned_xform = np.eye(4)
        learned_xform[:3, 3] = -learned_xforms[0].cpu().numpy()[i]
        gt_xform = np.eye(4)
        gt_xform[:3, 3] = -gt_xforms[i]
        # learned_ext = learned_geom[i]
        learned_ext = scales[i] * gt_exts.numpy()[i]
        ext_xform = (learned_ext, learned_xform)
        # ext_xform = (ext, learned_xform)
        prex_ext_xform = (ext, gt_xform)
        obbs_of_interest[i] = [ext_xform]
        prex_obbs_of_interest[i] = [prex_ext_xform]

    # normal
    query_points = torch.zeros((1, num_parts, pt_sample_res**3, 3)).to(torch.float32)
    for i in range(num_parts):
        part_query_points = reconstruct.make_query_points(
            pt_sample_res,
            limits=[(-0.5/scales[i, 0], 0.5/scales[i, 0]),
                    (-0.5/scales[i, 1], 0.5/scales[i, 1]),
                    (-0.5/scales[i, 2], 0.5/scales[i, 2])])
        query_points[:, i] = torch.from_numpy(part_query_points)
    query_points = query_points.to(device)

    with torch.no_grad():
        transformed_points = query_points + learned_xforms.unsqueeze(2)
        occs2 = model(transformed_points.masked_fill(points_mask==1,torch.tensor(0)),
                      batch_embed.masked_fill(parts_mask==1, torch.tensor(0)))
        occs2 = occs2.masked_fill(occ_mask==1,torch.tensor(float('-inf')))
        pred_values2, _ = torch.max(occs2, dim=-1, keepdim=True)

    if args.mask:
        mag = 0.8
    else:
        mag = 0.8

    sdf_grid = torch.reshape(
        pred_values1,
        (1, pt_sample_res, pt_sample_res, pt_sample_res))
    sdf_grid = torch.permute(sdf_grid, (0, 2, 1, 3))
    np.save(os.path.join(results_dir, 'prex_occ.npy'),
            sdf_grid.cpu().numpy())
    vertices, faces =\
        kaolin.ops.conversions.voxelgrids_to_trianglemeshes(sdf_grid)
    vertices = kaolin.ops.pointcloud.center_points(
        vertices[0].unsqueeze(0), normalize=True).squeeze(0)
    pred_vertices = vertices.cpu().numpy()
    pred_faces = faces[0].cpu().numpy()
    prex_pred_mesh = trimesh.Trimesh(pred_vertices, pred_faces)
    prex_pred_mesh.export(os.path.join(results_dir, 'prex_mesh_pred.obj'))
    visualize.save_mesh_vis(prex_pred_mesh, prex_pred_mesh_path,
                            mag=mag, white_bg=True)
    
    sdf_grid = torch.reshape(
        pred_values2,
        (1, pt_sample_res, pt_sample_res, pt_sample_res))
    sdf_grid = torch.permute(sdf_grid, (0, 2, 1, 3))
    np.save(os.path.join(results_dir, 'occ.npy'),
            sdf_grid.cpu().numpy())
    vertices, faces =\
        kaolin.ops.conversions.voxelgrids_to_trianglemeshes(sdf_grid)
    vertices = kaolin.ops.pointcloud.center_points(
        vertices[0].unsqueeze(0), normalize=True).squeeze(0)
    pred_vertices = vertices.cpu().numpy()
    pred_faces = faces[0].cpu().numpy()
    pred_mesh = trimesh.Trimesh(pred_vertices, pred_faces)
    pred_mesh.export(os.path.join(results_dir, 'mesh_pred.obj'))
    visualize.save_mesh_vis(pred_mesh, pred_mesh_path,
                            mag=mag, white_bg=True)
    
    # exit(0)
    masked_indices = masked_indices.cpu().numpy()
    unmasked_indices = list(set(range(num_parts)) - set(masked_indices))

    import itertools
    obbs_of_interest = [obbs_of_interest[x] for x in unmasked_indices]
    obbs_of_interest = list(itertools.chain(*obbs_of_interest))
    visualize.save_obbs_vis(obbs_of_interest,
                            obbs_path, mag=mag, white_bg=True,
                            unmasked_indices=unmasked_indices)
    
    prex_obbs_of_interest = [prex_obbs_of_interest[x] for x in unmasked_indices]
    prex_obbs_of_interest = list(itertools.chain(*prex_obbs_of_interest))
    visualize.save_obbs_vis(prex_obbs_of_interest,
                            prex_obbs_path, mag=mag, white_bg=True,
                            unmasked_indices=unmasked_indices)
    
    if not args.mask:
        visualize.stitch_imges(
            os.path.join(results_dir,f'assembly_results.png'),
            image_paths=lst_paths,
            adj=100)
    else:
        # parts_str contains all the parts that are MASKED OUT
        parts_str = '-'.join([str(x) for x in masked_indices])
        visualize.stitch_imges(
            os.path.join(results_dir,f'assembly_results_mask_{parts_str}.png'),
            image_paths=lst_paths,
            adj=100)
        
    exit(0)


if args.post_process:
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
        batch_embed = embeddings(torch.tensor(model_idx).to(device)).unsqueeze(0).detach()
    else:
        batch_embed = embeddings(torch.tensor(0).to(device)).unsqueeze(0).detach()

    unique_part_names, name_to_ori_ids_and_objs,\
        orig_obbs, entire_mesh, name_to_obbs, _, _, _ =\
        preprocess_data_17.merge_partnet_after_merging(anno_id)

    with open(f'data/{cat_name}_part_name_to_new_id_17_{ds_start}_{ds_end}.json', 'r') as f:
        unique_name_to_new_id = json.load(f)

    gt_color = [31, 119, 180, 255]
    mesh_pred_path = os.path.join(results_dir, 'mesh_pred.png')
    mesh_gt_path = os.path.join(results_dir, 'mesh_gt.png')
    learned_obbs_path = os.path.join(results_dir, 'obbs_pred.png')
    obbs_path = os.path.join(results_dir, 'obbs_gt.png')
    lst_paths = [
        obbs_path,
        mesh_gt_path,
        learned_obbs_path,
        mesh_pred_path]

    recon_res = 32
    query_points = reconstruct.make_query_points(recon_res)
    query_points = torch.from_numpy(query_points).to(device, torch.float32)
    query_points = query_points.unsqueeze(0)
    bs, num_points, _ = query_points.shape

    if not OVERFIT:
        batch_node_feat = torch.from_numpy(node_features[model_idx:model_idx+1, :, :3]).to(device, torch.float32)
        batch_adj = torch.from_numpy(adj[model_idx:model_idx+1]).to(device, torch.float32)
        batch_part_nodes = torch.from_numpy(part_nodes[model_idx:model_idx+1]).to(device, torch.float32)

    with torch.no_grad():
        if args.mask:
            # parts = [args.part]
            parts = args.parts
            parts = [int(x) for x in parts]
            if args.recon_one_part:
                # only reconstruct part specified by parts
                masked_indices = list(set(range(num_parts)) - set(parts))
                masked_indices = torch.Tensor(masked_indices).to(device, torch.long)
            if args.recon_the_rest:
                # don't reconstruct stuff in masked_indices
                masked_indices = torch.Tensor(parts).to(device, torch.long)
        else:
            masked_indices = torch.Tensor([]).to(device, torch.long)

        if not args.mask:
            print("reconstructing...")
        else:
            print(f"masking parts: {masked_indices.cpu().numpy().tolist()}")

        parts_mask = torch.zeros((batch_embed.shape[0], num_parts)).to(device, torch.float32)
        parts_mask[:, masked_indices] = 1
        parts_mask = torch.repeat_interleave(parts_mask, each_part_feat, dim=-1)
        
        points_mask = torch.zeros((1, num_parts, 1, 1)).to(device, torch.float32)
        points_mask[:, masked_indices] = 1

        occ_mask = torch.zeros((1, 1, num_parts)).to(device, torch.float32)
        occ_mask[:, :, masked_indices] = 1

        # learning xforms
        batch_vec = torch.arange(start=0, end=1).to(device)
        batch_vec = torch.repeat_interleave(batch_vec, num_union_nodes)
        batch_mask = torch.sum(batch_part_nodes, dim=1)
        learned_geom, learned_xforms = model.learn_geom_xform(batch_node_feat,
                                                    batch_adj,
                                                    batch_mask,
                                                    batch_vec)
        learned_relations = learned_xforms[:, connectivity[:, 0], connectivity[:, 1], :]
        learned_xforms = learned_xforms[:, [0, 1, 2, 3], [0, 1, 2, 3], :]

        batch_geom = torch.einsum('ijk, ikm -> ijm',
                                batch_part_nodes.to(torch.float32),
                                batch_node_feat)

        pairwise_xforms = learned_xforms[:, connectivity]
        learned_relations = pairwise_xforms[:, :, 1] - pairwise_xforms[:, :, 0]

        transformed_points = query_points.unsqueeze(1).expand(-1, num_parts, -1, -1) +\
            learned_xforms.unsqueeze(2)

        occs1 = model(transformed_points.masked_fill(points_mask==1,torch.tensor(0)),
                      batch_embed.masked_fill(parts_mask==1, torch.tensor(0)))
        occs1 = occs1.masked_fill(occ_mask==1,torch.tensor(float('-inf')))
        pred_values, _ = torch.max(occs1, dim=-1, keepdim=True)
    
    if args.mask:
        mag = 1
    else:
        mag = 0.8

    sdf_grid = torch.reshape(
        pred_values,
        (1, recon_res, recon_res, recon_res))
    sdf_grid = torch.permute(sdf_grid, (0, 2, 1, 3))
    vertices, faces =\
        kaolin.ops.conversions.voxelgrids_to_trianglemeshes(sdf_grid)
    vertices = kaolin.ops.pointcloud.center_points(
        vertices[0].unsqueeze(0), normalize=True).squeeze(0)
    pred_vertices = vertices.cpu().numpy()
    pred_faces = faces[0].cpu().numpy()
    pred_mesh = trimesh.Trimesh(pred_vertices, pred_faces)
    pred_mesh.export(os.path.join(results_dir, 'mesh_pred.obj'))
    visualize.save_mesh_vis(pred_mesh, mesh_pred_path,
                            mag=mag, white_bg=white_bg)

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
    aligned_gt_mesh.export(os.path.join(results_dir, 'mesh_gt.obj'))
    aligned_gt_mesh.visual.vertex_colors = [31, 119, 180]
    visualize.save_mesh_vis(aligned_gt_mesh, mesh_gt_path,
                            mag=mag, white_bg=white_bg)
    
    timelapse = kaolin.visualize.Timelapse(os.path.join(results_dir, 'timelapse'))
    timelapse.add_mesh_batch(category='gt_mesh',
                             vertices_list=[torch.from_numpy(aligned_gt_mesh.vertices)],
                             faces_list=[torch.from_numpy(aligned_gt_mesh.faces)])
    
    timelapse.add_mesh_batch(category='pred_coarse_mesh',
                             vertices_list=[torch.from_numpy(pred_mesh.vertices)],
                             faces_list=[torch.from_numpy(pred_mesh.faces)])
    
    pcd, _ = trimesh.sample.sample_surface(aligned_gt_mesh, 10000, seed=319)
    pcd = torch.from_numpy(pcd).to(device, torch.float32)
    # pcd = kaolin.ops.pointcloud.center_points(pcd.unsqueeze(0), normalize=True).squeeze(0) * 0.9
    # timelapse.add_pointcloud_batch(category='input',
    #                                pointcloud_list=[pcd.cpu()], points_type = "usd_geom_points")
    
    # load tets
    grid_res = 128
    tet_verts = torch.tensor(
        np.load('samples/{}_verts.npz'.format(grid_res))['data'],
        dtype=torch.float, device=device)
    tets = torch.tensor(
        ([np.load('samples/{}_tets_{}.npz'.format(grid_res, i))['data'] for i in range(4)]),
        dtype=torch.long, device=device).permute(1,0)
    
    from dmtet_networks import dmtet_utils, dmtet_network

    dmtet_model = dmtet_network.Decoder(multires=multires).to(device)
    # dmtet_model = dmtet_network.Decoder(input_dims=3+each_part_feat*4,
    #                                     multires=multires).to(device)
    sdf = - 2 * pred_values + 1
    # sdf = 1 - pred_values
    # sdf /= torch.sum(sdf)
    dmtet_model.pre_train_sphere(10000)
    # dmtet_model.pre_train_chair(10000,
    #                             query_points.squeeze(0),
    #                             sdf.squeeze(0).squeeze(-1))
    # dmtet_model.pre_train_sphere_feat(10000, batch_embed)
    # dmtet_model.pre_train_chair_feat(10000,
    #                                  query_points.squeeze(0),
    #                                  sdf.squeeze(0).squeeze(-1),
    #                                  batch_embed)

    vars = [p for _, p in dmtet_model.named_parameters()]
    lr = 5e-5
    optimizer = torch.optim.Adam(vars, lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.0, 10**(-x*0.0002))) # LR decay over time

    iterations = 10000
    for it in range(iterations):
        pred = dmtet_model(tet_verts) # predict SDF and per-vertex deformation
        # pred = dmtet_model.forward_feat(tet_verts, batch_embed) # predict SDF and per-vertex deformation
        sdf, deform = pred[:,0], pred[:,1:]
        verts_deformed = tet_verts + torch.tanh(deform) / grid_res # constraint deformation to avoid flipping tets
        mesh_verts, mesh_faces = kaolin.ops.conversions.marching_tetrahedra(
            verts_deformed.unsqueeze(0), tets, sdf.unsqueeze(0)) # running MT (batched) to extract surface mesh
        mesh_verts, mesh_faces = mesh_verts[0], mesh_faces[0]

        loss = dmtet_utils.loss_f(iterations, mesh_verts, mesh_faces, pcd, it)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (it) % 100 == 0 or it == (iterations - 1): 
            print ('Iteration {} - loss: {:.5f}, # of mesh vertices: {}, # of mesh faces: {}'.format(it, loss, mesh_verts.shape[0], mesh_faces.shape[0]))
            # save reconstructed mesh
            timelapse.add_mesh_batch(
                iteration=it+1,
                category='extracted_mesh',
                vertices_list=[mesh_verts.cpu()],
                faces_list=[mesh_faces.cpu()]
            )

    exit(0)


if args.post_process_fc:
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
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    results_dir = os.path.join(results_dir, anno_id)
    misc.check_dir(results_dir)

    if not OVERFIT:
        # part_nodes = part_nodes[model_idx].unsqueeze(0)
        batch_embed = embeddings(torch.tensor(model_idx).to(device)).unsqueeze(0).detach()
    else:
        batch_embed = embeddings(torch.tensor(0).to(device)).unsqueeze(0).detach()

    unique_part_names, name_to_ori_ids_and_objs,\
        orig_obbs, entire_mesh, name_to_obbs, _, _, _ =\
        preprocess_data_17.merge_partnet_after_merging(anno_id)

    with open(f'data/{cat_name}_part_name_to_new_id_17_{ds_start}_{ds_end}.json', 'r') as f:
        unique_name_to_new_id = json.load(f)

    gt_color = [31, 119, 180, 255]
    mesh_pred_path = os.path.join(results_dir, 'mesh_pred.png')
    mesh_gt_path = os.path.join(results_dir, 'mesh_gt.png')
    learned_obbs_path = os.path.join(results_dir, 'obbs_pred.png')
    obbs_path = os.path.join(results_dir, 'obbs_gt.png')
    lst_paths = [
        obbs_path,
        mesh_gt_path,
        learned_obbs_path,
        mesh_pred_path]

    recon_res = 32
    query_points = reconstruct.make_query_points(recon_res)
    query_points = torch.from_numpy(query_points).to(device, torch.float32)
    query_points = query_points.unsqueeze(0)
    bs, num_points, _ = query_points.shape

    if not OVERFIT:
        batch_node_feat = torch.from_numpy(node_features[model_idx:model_idx+1, :, :3]).to(device, torch.float32)
        batch_adj = torch.from_numpy(adj[model_idx:model_idx+1]).to(device, torch.float32)
        batch_part_nodes = torch.from_numpy(part_nodes[model_idx:model_idx+1]).to(device, torch.float32)

    with torch.no_grad():
        if args.mask:
            # parts = [args.part]
            parts = args.parts
            parts = [int(x) for x in parts]
            if args.recon_one_part:
                # only reconstruct part specified by parts
                masked_indices = list(set(range(num_parts)) - set(parts))
                masked_indices = torch.Tensor(masked_indices).to(device, torch.long)
            if args.recon_the_rest:
                # don't reconstruct stuff in masked_indices
                masked_indices = torch.Tensor(parts).to(device, torch.long)
        else:
            masked_indices = torch.Tensor([]).to(device, torch.long)

        if not args.mask:
            print("reconstructing...")
        else:
            print(f"masking parts: {masked_indices.cpu().numpy().tolist()}")

        parts_mask = torch.zeros((batch_embed.shape[0], num_parts)).to(device, torch.float32)
        parts_mask[:, masked_indices] = 1
        parts_mask = torch.repeat_interleave(parts_mask, each_part_feat, dim=-1)
        
        points_mask = torch.zeros((1, num_parts, 1, 1)).to(device, torch.float32)
        points_mask[:, masked_indices] = 1

        occ_mask = torch.zeros((1, 1, num_parts)).to(device, torch.float32)
        occ_mask[:, :, masked_indices] = 1

        # learning xforms
        batch_vec = torch.arange(start=0, end=1).to(device)
        batch_vec = torch.repeat_interleave(batch_vec, num_union_nodes)
        batch_mask = torch.sum(batch_part_nodes, dim=1)
        learned_geom, learned_xforms = model.learn_geom_xform(batch_node_feat,
                                                    batch_adj,
                                                    batch_mask,
                                                    batch_vec)
        learned_relations = learned_xforms[:, connectivity[:, 0], connectivity[:, 1], :]
        learned_xforms = learned_xforms[:, [0, 1, 2, 3], [0, 1, 2, 3], :]

        batch_geom = torch.einsum('ijk, ikm -> ijm',
                                batch_part_nodes.to(torch.float32),
                                batch_node_feat)

        pairwise_xforms = learned_xforms[:, connectivity]
        learned_relations = pairwise_xforms[:, :, 1] - pairwise_xforms[:, :, 0]

        transformed_points = query_points.unsqueeze(1).expand(-1, num_parts, -1, -1) +\
            learned_xforms.unsqueeze(2)

        occs1 = model(transformed_points.masked_fill(points_mask==1,torch.tensor(0)),
                      batch_embed.masked_fill(parts_mask==1, torch.tensor(0)))
        occs1 = occs1.masked_fill(occ_mask==1,torch.tensor(float('-inf')))
        pred_values, _ = torch.max(occs1, dim=-1, keepdim=True)
    
    if args.mask:
        mag = 1
    else:
        mag = 0.8

    sdf_grid = torch.reshape(
        pred_values,
        (1, recon_res, recon_res, recon_res))
    sdf_grid = torch.permute(sdf_grid, (0, 2, 1, 3))
    vertices, faces =\
        kaolin.ops.conversions.voxelgrids_to_trianglemeshes(sdf_grid)
    vertices = kaolin.ops.pointcloud.center_points(
        vertices[0].unsqueeze(0), normalize=True).squeeze(0)
    pred_vertices = vertices.cpu().numpy()
    pred_faces = faces[0].cpu().numpy()
    pred_mesh = trimesh.Trimesh(pred_vertices, pred_faces)
    pred_mesh.export(os.path.join(results_dir, 'mesh_pred.obj'))
    # visualize.save_mesh_vis(pred_mesh, mesh_pred_path,
    #                         mag=mag, white_bg=white_bg)

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
    aligned_gt_mesh_path = os.path.join(results_dir, 'mesh_gt.obj')
    aligned_gt_mesh.export(aligned_gt_mesh_path)
    aligned_gt_mesh.visual.vertex_colors = [31, 119, 180]
    # visualize.save_mesh_vis(aligned_gt_mesh, mesh_gt_path,
    #                         mag=mag, white_bg=white_bg)
    
    timelapse = kaolin.visualize.Timelapse(os.path.join(results_dir, 'timelapse_flexi'))
    timelapse.add_mesh_batch(category='gt_mesh',
                             vertices_list=[torch.from_numpy(aligned_gt_mesh.vertices)],
                             faces_list=[torch.from_numpy(aligned_gt_mesh.faces)])
    
    timelapse.add_mesh_batch(category='pred_coarse_mesh',
                             vertices_list=[torch.from_numpy(pred_mesh.vertices)],
                             faces_list=[torch.from_numpy(pred_mesh.faces)])
    
    # pcd, _ = trimesh.sample.sample_surface(aligned_gt_mesh, 10000, seed=319)
    # pcd = torch.from_numpy(pcd).to(device, torch.float32)

    from flexi.flexicubes import FlexiCubes
    from flexi.util import *
    from flexi import render, loss
    import imageio

    glctx = dr.RasterizeGLContext()
    gt_mesh = load_mesh(aligned_gt_mesh_path, device)
    gt_mesh.auto_normals()

    voxel_grid_res = 31

    fc = FlexiCubes(device)
    x_nx3_orig, cube_fx8 = fc.construct_voxel_grid(voxel_grid_res)
    x_nx3 = 2*x_nx3_orig # scale up the grid so that it's larger than the target object
    # x_nx3 = x_nx3_orig

    with torch.no_grad():
        if args.mask:
            # parts = [args.part]
            parts = args.parts
            parts = [int(x) for x in parts]
            if args.recon_one_part:
                # only reconstruct part specified by parts
                masked_indices = list(set(range(num_parts)) - set(parts))
                masked_indices = torch.Tensor(masked_indices).to(device, torch.long)
            if args.recon_the_rest:
                # don't reconstruct stuff in masked_indices
                masked_indices = torch.Tensor(parts).to(device, torch.long)
        else:
            masked_indices = torch.Tensor([]).to(device, torch.long)

        if not args.mask:
            print("reconstructing...")
        else:
            print(f"masking parts: {masked_indices.cpu().numpy().tolist()}")

        parts_mask = torch.zeros((batch_embed.shape[0], num_parts)).to(device, torch.float32)
        parts_mask[:, masked_indices] = 1
        parts_mask = torch.repeat_interleave(parts_mask, each_part_feat, dim=-1)
        
        points_mask = torch.zeros((1, num_parts, 1, 1)).to(device, torch.float32)
        points_mask[:, masked_indices] = 1

        occ_mask = torch.zeros((1, 1, num_parts)).to(device, torch.float32)
        occ_mask[:, :, masked_indices] = 1

        # learning xforms
        batch_vec = torch.arange(start=0, end=1).to(device)
        batch_vec = torch.repeat_interleave(batch_vec, num_union_nodes)
        batch_mask = torch.sum(batch_part_nodes, dim=1)
        learned_geom, learned_xforms = model.learn_geom_xform(batch_node_feat,
                                                    batch_adj,
                                                    batch_mask,
                                                    batch_vec)
        learned_relations = learned_xforms[:, connectivity[:, 0], connectivity[:, 1], :]
        learned_xforms = learned_xforms[:, [0, 1, 2, 3], [0, 1, 2, 3], :]

        batch_geom = torch.einsum('ijk, ikm -> ijm',
                                batch_part_nodes.to(torch.float32),
                                batch_node_feat)

        pairwise_xforms = learned_xforms[:, connectivity]
        learned_relations = pairwise_xforms[:, :, 1] - pairwise_xforms[:, :, 0]

        transformed_points = x_nx3_orig.unsqueeze(0).unsqueeze(1).expand(-1, num_parts, -1, -1) +\
            learned_xforms.unsqueeze(2)

        occs1 = model(transformed_points.masked_fill(points_mask==1,torch.tensor(0)),
                      batch_embed.masked_fill(parts_mask==1, torch.tensor(0)))
        occs1 = occs1.masked_fill(occ_mask==1,torch.tensor(float('-inf')))
        pred_values, _ = torch.max(occs1, dim=-1, keepdim=True)
    
    # sdf = torch.rand_like(x_nx3[:,0]) - 0.1 # randomly init SDF
    
    sdf = - 2 * pred_values + 1
    sdf = torch.squeeze(sdf)

    orig_sdf = sdf.clone().detach()
    # orig_sdf.clone().detach()

    # sdf    = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
    # sdf    = sdf.clone().detach()

    # set per-cube learnable weights to zeros
    weight = torch.zeros((cube_fx8.shape[0], 21), dtype=torch.float, device='cuda') 
    weight    = torch.nn.Parameter(weight.clone().detach(), requires_grad=True)
    deform = torch.nn.Parameter(torch.zeros_like(x_nx3), requires_grad=True)
    # deform = torch.zeros_like(x_nx3)
    
    class PPNet(torch.nn.Module):
        def __init__(self, feature_dims=32*4, output_dims=3, hidden=0,
                     internal_dims=64):
            super().__init__()
            # internal_dims = 64
            use_bias = True

            net = (torch.nn.Linear(feature_dims, internal_dims, bias=use_bias),
                   torch.nn.ReLU())
            for i in range(hidden-1):
                net = net + (
                    torch.nn.Linear(internal_dims, internal_dims, bias=use_bias),
                    torch.nn.ReLU())
            net = net + (torch.nn.Linear(internal_dims, output_dims, bias=False),)
            self.net = torch.nn.Sequential(*net)

        def forward(self, feature, final_layer=False):
            x = self.net(feature)
            if final_layer:
                x = torch.nn.Tanh()(x)
            return x
    
    batch_embed_sdf_deform = batch_embed.expand(x_nx3.shape[0], -1)
    batch_embed_weight = batch_embed.expand(cube_fx8.shape[0], -1)

    sdf_model = PPNet(3 + num_parts*each_part_feat, output_dims=1, hidden=5,
                      internal_dims=128).to(device)
    sdf_params = [p for _, p in sdf_model.named_parameters()]

    sdf_embed = torch.nn.Embedding(1,
                                   num_parts*each_part_feat).to(device)
    torch.nn.init.normal_(
        sdf_embed.weight.data,
        0.0,
        1 / math.sqrt(num_parts*each_part_feat),
    )
    batch_sdf_embed = sdf_embed.weight.data.expand(x_nx3.shape[0], -1)

    # sdf_in = torch.concat([x_nx3_orig,
    #                       batch_embed_sdf_deform], dim=-1)
    sdf_in = torch.concat([x_nx3_orig,
                           batch_sdf_embed], dim=-1)

    # # pretrain SDF model
    # # pred_values = torch.squeeze(pred_values)
    # pre_t_optimizer = torch.optim.Adam(
    #     list(sdf_model.parameters()) + list(sdf_embed.parameters()), lr=1e-4)
    # pre_t_loss_fn = torch.nn.MSELoss()
    # for i in tqdm(range(5000)):
    #     output = sdf_model(sdf_in
    #                        , final_layer=True
    #                        )
    #     # output = sdf_model(batch_embed_sdf_deform)
    #     pre_t_loss = pre_t_loss_fn(output[...,0] + orig_sdf, sdf)
    #     pre_t_optimizer.zero_grad()
    #     pre_t_loss.backward()
    #     pre_t_optimizer.step()
    # print("Pre-trained SDF", pre_t_loss.item())

    # exit(0)

    weight_model = PPNet(num_parts*each_part_feat, output_dims=21, hidden=0).to(device)
    weight_params = [p for _, p in weight_model.named_parameters()]

    deform_model = PPNet(num_parts*each_part_feat, output_dims=3, hidden=1).to(device)
    deform_params = [p for _, p in deform_model.named_parameters()]

    # batch_embed.requires_grad_()
    # print(batch_embed.requires_grad)
    # exit(0)

    #  Retrieve all the edges of the voxel grid; these edges will be utilized to 
    #  compute the regularization loss in subsequent steps of the process.    
    all_edges = cube_fx8[:, fc.cube_edges].reshape(-1, 2)
    grid_edges = torch.unique(all_edges, dim=0)

    learning_rate = 0.01
    sdf_regularizer = 0.2

    def lr_schedule(iter):
        return max(0.0, 10**(-(iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.    

    # optimizer = torch.optim.Adam([sdf, weight, deform], lr=learning_rate)
    # optimizer = torch.optim.Adam([sdf, deform], lr=learning_rate)
    # optimizer = torch.optim.Adam([sdf] + deform_params, lr=learning_rate)
    # optimizer = torch.optim.Adam(sdf_params + deform_params, lr=learning_rate)
    # optimizer = torch.optim.Adam(deform_params, lr=learning_rate)
    # optimizer = torch.optim.Adam([{"sdf": sdf, "weight": weight, "params": params}], lr=learning_rate)
    # optimizer = torch.optim.Adam([sdf, weight] + deform_params, lr=learning_rate)
    # optimizer = torch.optim.Adam([sdf] + weight_params + deform_params, lr=learning_rate)
    # optimizer = torch.optim.Adam([weight] + sdf_params + deform_params, lr=learning_rate)
    # optimizer = torch.optim.Adam([sdf, weight], lr=learning_rate)
    # optimizer = torch.optim.Adam([weight], lr=learning_rate)
    # optimizer = torch.optim.Adam([{"params": weight_params, "lr": learning_rate},
    #                               {"params": deform_params, "lr": learning_rate}])
    # optimizer = torch.optim.Adam([{"params": weight_params, "lr": learning_rate},
    #                               {"params": deform_params, "lr": learning_rate},
    #                               {"params": sdf_params, "lr": learning_rate}])
    # optimizer = torch.optim.Adam(weight_params + sdf_params + deform_params, lr=learning_rate)
    optimizer = torch.optim.Adam([{"params": sdf_params, "lr": lr},
                                  {"params": sdf_embed.parameters(), "lr": lr},
                                  {"params": deform_params, "lr": lr}])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x)) 
    
    iterations = 1000

    train_res = [2048, 2048]
    display_res = [512, 512]
    train_img_dir = os.path.join(results_dir, 'flexi_out')
    misc.check_dir(train_img_dir)

    print("second pass with FlexiCubes...")
    for it in range(iterations):
        # print(it)
        optimizer.zero_grad()
        # sample random camera poses
        mv, mvp = render.get_random_camera_batch(8, iter_res=train_res, device=device, use_kaolin=False)
        # render gt mesh
        target = render.render_mesh_paper(gt_mesh, mv, mvp, train_res)
        
        # learn deform from code
        deform = deform_model(batch_embed_sdf_deform)
        
        # extract and render FlexiCubes mesh
        grid_verts = x_nx3 + (2-1e-8) / (voxel_grid_res * 2) * torch.tanh(deform)
        
        # learn sdf from code
        # sdf = sdf_model(sdf_in
        #                 # , final_layer=True
        #                 )
        delta_sdf = sdf_model(sdf_in
                              , final_layer=True
                             )
        sdf = orig_sdf.unsqueeze(-1) + delta_sdf
        # print(torch.min(sdf), torch.max(sdf))
        # print(torch.min(delta_sdf), torch.max(delta_sdf))
        
        # learn weight from code
        # weight = weight_model(batch_embed_weight)
        
        # vertices, faces, L_dev = fc(grid_verts, sdf, cube_fx8, voxel_grid_res, beta_fx12=weight[:,:12], alpha_fx8=weight[:,12:20],
        #     gamma_f=weight[:,20], training=True)
        vertices, faces, L_dev = fc(grid_verts, sdf, cube_fx8, voxel_grid_res, training=True)
        # print(faces)
        flexicubes_mesh = Mesh(vertices, faces)
        buffers = render.render_mesh_paper(flexicubes_mesh, mv, mvp, train_res)
        
        # evaluate reconstruction loss
        mask_loss = (buffers['mask'] - target['mask']).abs().mean()
        depth_loss = (((((buffers['depth'] - (target['depth']))* target['mask'])**2).sum(-1)+1e-8)).sqrt().mean() * 10
    
        t_iter = it / iterations
        sdf_weight = sdf_regularizer - (sdf_regularizer - sdf_regularizer/20)*min(1.0, 4.0 * t_iter)
        reg_loss = loss.sdf_reg_loss(sdf, grid_edges).mean() * sdf_weight # Loss to eliminate internal floaters that are not visible
        reg_loss += L_dev.mean() * 0.5
        reg_loss += (weight[:,:20]).abs().mean() * 0.1
        total_loss = mask_loss + depth_loss + reg_loss

        # with torch.no_grad():
        #     pts = sample_random_points(1000, gt_mesh)
        #     gt_sdf = compute_sdf(pts, gt_mesh.vertices, gt_mesh.faces)
        # pred_sdf = compute_sdf(pts, flexicubes_mesh.vertices, flexicubes_mesh.faces)
        # total_loss += torch.nn.functional.mse_loss(pred_sdf, gt_sdf) * 2e3
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if (it) % 100 == 0 or it == (iterations - 1): 
            with torch.no_grad():
                # extract mesh with training=False
                # vertices, faces, L_dev = fc(
                #     grid_verts, sdf, cube_fx8, voxel_grid_res,
                #     beta_fx12=weight[:,:12], alpha_fx8=weight[:,12:20],
                #     gamma_f=weight[:,20], training=False)
                vertices, faces, L_dev = fc(
                    grid_verts, sdf, cube_fx8, voxel_grid_res,
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

        # save_every = 20
        # if (it % save_every == 0 or it == (iterations-1)): # save normal image for visualization
        #     with torch.no_grad():
        #         # extract mesh with training=False
        #         vertices, faces, L_dev = fc(grid_verts, sdf, cube_fx8, voxel_grid_res, beta_fx12=weight[:,:12], alpha_fx8=weight[:,12:20],
        #         gamma_f=weight[:,20], training=False)
        #         flexicubes_mesh = Mesh(vertices, faces)
                
        #         flexicubes_mesh.auto_normals() # compute face normals for visualization
        #         mv, mvp = render.get_rotate_camera(it//save_every, iter_res=display_res, device=device,use_kaolin=False)
        #         val_buffers = render.render_mesh_paper(flexicubes_mesh, mv.unsqueeze(0), mvp.unsqueeze(0), display_res, return_types=["normal"], white_bg=True)
        #         val_image = ((val_buffers["normal"][0].detach().cpu().numpy()+1)/2*255).astype(np.uint8)
                
        #         gt_buffers = render.render_mesh_paper(gt_mesh, mv.unsqueeze(0), mvp.unsqueeze(0), display_res, return_types=["normal"], white_bg=True)
        #         gt_image = ((gt_buffers["normal"][0].detach().cpu().numpy()+1)/2*255).astype(np.uint8)
        #         imageio.imwrite(os.path.join(train_img_dir, '{:04d}.png'.format(it)), np.concatenate([val_image, gt_image], 1))
        #         print(f"Optimization Step [{it}/{iterations}], Loss: {total_loss.item():.4f}")
        

    mesh_np = trimesh.Trimesh(vertices = vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), process=False)
    mesh_np.export(os.path.join(results_dir, 'flexi_mesh.obj'))
# 