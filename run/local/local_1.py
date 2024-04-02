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
from occ_networks.basic_decoder_nasa import SDFDecoder
from utils import misc, visualize, transform, ops, reconstruct
from data_prep import preprocess_data_0
from typing import Dict, List
from anytree.exporter import UniqueDotExporter


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
parser.add_argument('--of', action="store_true", default=False)
parser.add_argument('--of_idx', '--oi', type=int)
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
lr = 3e-3
laplacian_weight = 0.1
iterations = 3001
save_every = 100
multires = 2
pt_sample_res = 64        # point_sampling

expt_id = 1

OVERFIT = args.of
overfit_idx = args.of_idx

batch_size = 10
if OVERFIT:
    batch_size = 1

ds_start = 0
ds_end = 170

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
# cat_id = '03636649'

train_new_ids_to_objs_path = f'data/{cat_name}_train_new_ids_to_objs_0_{ds_start}_{ds_end}.json'
with open(train_new_ids_to_objs_path, 'r') as f:
    train_new_ids_to_objs: Dict = json.load(f)
model_idx_to_anno_id = {}
for model_idx, anno_id in enumerate(train_new_ids_to_objs.keys()):
    model_idx_to_anno_id[model_idx] = anno_id

train_data_path = f'data/{cat_name}_train_{pt_sample_res}_0_{ds_start}_{ds_end}.hdf5'
train_data = h5py.File(train_data_path, 'r')
if OVERFIT:
    part_num_indices = train_data['part_num_indices'][overfit_idx:overfit_idx+1]
    all_indices = train_data['all_indices'][overfit_idx:overfit_idx+1]
    # normalized_points = train_data['normalized_points'][overfit_idx:overfit_idx+1]
    values = train_data['values'][overfit_idx:overfit_idx+1]
    part_nodes = train_data['part_nodes'][overfit_idx:overfit_idx+1]
    xforms = train_data['xforms'][overfit_idx:overfit_idx+1]
    transformed_points = train_data['transformed_points'][overfit_idx:overfit_idx+1]
    empty_parts = train_data['empty_parts'][overfit_idx:overfit_idx+1]
else:
    part_num_indices = train_data['part_num_indices']
    all_indices = train_data['all_indices']
    # normalized_points = train_data['normalized_points']
    values = train_data['values']
    part_nodes = train_data['part_nodes']
    xforms = train_data['xforms']
    transformed_points = train_data['transformed_points']
    empty_parts = train_data['empty_parts']

num_points = transformed_points.shape[1]
num_shapes, num_parts = part_num_indices.shape
all_fg_part_indices = []
for i in range(num_shapes):
    indices = preprocess_data_0.convert_flat_list_to_fg_part_indices(
        part_num_indices[i], all_indices[i])
    all_fg_part_indices.append(np.array(indices, dtype=object))

n_batches = num_shapes // batch_size

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

# -------- model --------
each_part_feat = 8
model = SDFDecoder(num_parts=num_parts,
                   feature_dims=each_part_feat,
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

def load_batch(batch_idx, batch_size):
    start = batch_idx*batch_size
    end = start + batch_size
    return all_fg_part_indices[start:end],\
        torch.from_numpy(transformed_points[start:end]).to(device, torch.float32),\
        torch.from_numpy(values[start:end]).to(device, torch.float32),\
        None,\
        embeddings(torch.arange(start, end).to(device)),\
        torch.from_numpy(empty_parts[start:end]).to(device, torch.long)

optimizer = torch.optim.Adam([{"params": params, "lr": lr},
                              {"params": embeddings.parameters(), "lr": lr}])
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda x: max(0.0, 10**(-x*0.0002)))

def loss_f(pred_values, gt_values):
    recon_loss = mse(pred_values, gt_values)
    return recon_loss

torch.manual_seed(319)
np.random.seed(319)

def train_one_itr(it, all_fg_part_indices, 
                  transformed_points, values,
                  part_nodes, batch_embed, batch_empty_parts):
    num_parts_to_mask = np.random.randint(1, num_parts)
    # num_parts_to_mask = 1
    rand_indices = np.random.choice(num_parts, num_parts_to_mask,
                                    replace=False)

    masked_indices = torch.from_numpy(rand_indices).to(device, torch.long)
    # make gt value mask
    val_mask = torch.ones_like(values).to(device, torch.float32)
    for i in range(batch_size):
        fg_part_indices = all_fg_part_indices[i]
        fg_part_indices_masked = fg_part_indices[masked_indices.cpu().numpy()]
        if len(fg_part_indices_masked) != 0:
            fg_part_indices_masked = np.concatenate(fg_part_indices_masked, axis=0)
        else:
            fg_part_indices_masked = np.array([])
        val_mask[i][fg_part_indices_masked] = 0
    modified_values = values * val_mask

    parts_mask = torch.ones((batch_embed.shape[0], num_parts)).to(device, torch.float32)
    if len(masked_indices) != 0:
        parts_mask[:, rand_indices] = 0
    parts_mask = torch.repeat_interleave(parts_mask,
                                         each_part_feat, dim=-1)
    modified_embed = batch_embed * parts_mask

    points_mask = torch.ones_like(transformed_points).to(device, torch.float32)
    if len(masked_indices) != 0:
        points_mask[:, rand_indices] = 0

    occs1 = model(transformed_points * points_mask,
                  modified_embed)
    pred_values1, _ = torch.max(occs1, dim=-1, keepdim=True)
    loss1 = loss_f(pred_values1, modified_values)

    occs2 = model(transformed_points,
                  batch_embed)
    pred_values2, _ = torch.max(occs2, dim=-1, keepdim=True)
    loss2 = loss_f(pred_values2, values)

    loss = loss1 + loss2

    optimizer.zero_grad()
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
                batch_normalized_points, batch_values,\
                batch_part_nodes, batch_embed, batch_empty_parts =\
                    load_batch(b, batch_size)
            loss = train_one_itr(it,
                                 batch_fg_part_indices,
                                 batch_normalized_points,
                                 batch_values,
                                 batch_part_nodes,
                                 batch_embed,
                                 batch_empty_parts)
            batch_loss += loss
        avg_batch_loss = batch_loss / batch_size
            
        if (it) % 100 == 0 or it == (iterations - 1):
            info = f'Iteration {it} - loss: {avg_batch_loss:.8f}'
            print(info)

        if (it) % 1000 == 0 or it == (iterations - 1):
            torch.save({
                'epoch': it,
                'model_state_dict': model.state_dict(),
                'embeddings_state_dict': embeddings.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_batch_loss': avg_batch_loss,
                }, os.path.join(ckpt_dir, f'model_{it}.pt'))

    print("duration: ", time.time() - start_time)


def compute_iou(voxel_grid1, voxel_grid2):
    assert voxel_grid1.shape == voxel_grid2.shape,\
        "Voxel grids must have the same shape."
    intersection = torch.logical_and(voxel_grid1, voxel_grid2).sum()
    union = torch.logical_or(voxel_grid1, voxel_grid2).sum()
    iou = intersection / float(union) if union != 0 else 1.0
    return iou

# exit(0)

if args.test:
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
    embeddings = torch.nn.Embedding(num_shapes,
                                    num_parts*each_part_feat).to(device)
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
        orig_obbs, entire_mesh, name_to_obbs, obb_indices_to_ori_ids, _ =\
        preprocess_data_0.merge_partnet_after_merging(anno_id)

    with open(f'data/{cat_name}_part_name_to_new_id_0_{ds_start}_{ds_end}.json', 'r') as f:
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
    obbs_path = os.path.join(results_dir, 'obbs.png')
    lst_paths = [
        obbs_path,
        mesh_gt_path,
        mesh_pred_path]

    query_points = reconstruct.make_query_points(pt_sample_res)
    query_points = torch.from_numpy(query_points).to(device, torch.float32)
    query_points = query_points.unsqueeze(0)

    if not OVERFIT:
        xforms = xforms[model_idx:model_idx+1]

    bs, num_points, _ = query_points.shape
    transformed_points = torch.zeros((bs,
                                      num_parts,
                                      num_points,
                                      3)).to(device)    
    for i in range(num_parts):
        pts = query_points[0]
        xform = xforms[0, i]
        hom_pts = torch.cat([pts,
                            torch.ones((num_points, 1)).to(device)],
                            dim=-1).transpose(0, 1)
        xform_pts = torch.matmul(torch.tensor(xform).to(device), hom_pts).transpose(0, 1)[:, :3]
        transformed_points[0, i, :, :] = xform_pts

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

        parts_mask = torch.ones((batch_embed.shape[0], num_parts)).to(device, torch.float32)
        if len(masked_indices) != 0:
            parts_mask[:, masked_indices] = 0
        parts_mask = torch.repeat_interleave(parts_mask,
                                             each_part_feat, dim=-1)
        modified_embed = batch_embed * parts_mask

        points_mask = torch.ones_like(transformed_points).to(device, torch.float32)
        if len(masked_indices) != 0:
            points_mask[:, masked_indices] = 0

        pred_values = model(transformed_points * points_mask, modified_embed)
        pred_values, _ = torch.max(pred_values, dim=-1, keepdim=True)

    if args.mask:
        mag = 1
    else:
        mag = 0.6

    sdf_grid = torch.reshape(
        pred_values,
        (1, pt_sample_res, pt_sample_res, pt_sample_res))
    # print(f"sum: {torch.sum(sdf_grid).cpu()},\
    #       min: {torch.min(sdf_grid).cpu()},\
    #       max: {torch.max(sdf_grid).cpu()}")
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
                            mag=mag, white_bg=False)
    
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
                                mag=mag, white_bg=False)
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
                                mag=mag, white_bg=False)

    gt_samples, _ = trimesh.sample.sample_surface(
        aligned_gt_mesh, 10000, seed=319)
    gt_samples = torch.from_numpy(gt_samples).to(device)
    pred_samples, _ = trimesh.sample.sample_surface(
        pred_mesh, 10000, seed=319)
    pred_samples = torch.from_numpy(pred_samples).to(device)
    chamfer = kaolin.metrics.pointcloud.chamfer_distance(
        pred_samples.unsqueeze(0), gt_samples.unsqueeze(0)).mean()
    print("[EVAL] Chamfer distance: ", chamfer.cpu().numpy())

    obbs_of_interest = [part_obbs[x] for x in unmasked_indices]
    import itertools
    obbs_of_interest = list(itertools.chain(*obbs_of_interest))

    visualize.save_obbs_vis(obbs_of_interest,
                            obbs_path, mag=0.6, white_bg=False)
    
    if not args.mask:
        visualize.stitch_imges(
            os.path.join(results_dir,f'{anno_id}_results.png'),
            image_paths=lst_paths,
            adj=200)
    else:
        # parts_str contains all the parts that are MASKED OUT
        parts_str = '-'.join([str(x) for x in masked_indices])
        visualize.stitch_imges(
            os.path.join(results_dir,f'{anno_id}_results_mask_{parts_str}.png'),
            image_paths=lst_paths,
            adj=200)

    exit(0)

