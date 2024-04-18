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
from occ_networks.encoder_decoder_nasa_1 import VAE
from utils import misc, visualize, transform, ops, reconstruct
from data_prep import preprocess_data_8
from typing import Dict, List
from anytree.exporter import UniqueDotExporter
from data_prep.partnet_pts_dataset import PartPtsDataset
from torch_geometric.loader import DataLoader


parser = argparse.ArgumentParser()
# parser.add_argument('--id', type=str)
parser.add_argument('--train', action="store_true")
parser.add_argument('--test', action="store_true")
parser.add_argument('--test_idx', '--ti', type=int)
parser.add_argument('--it', type=int)
parser.add_argument('--of', action="store_true", default=False)
parser.add_argument('--of_idx', '--oi', type=int)
args = parser.parse_args()
# assert args.train != args.test, "Must pass in either train or test"
if args.test:
    assert args.it != None, "Must pass in ckpt iteration if testing"
if args.of:
    assert args.of_idx != None

device = 'cuda'
lr = 1e-3
laplacian_weight = 0.1
iterations = 10001
multires = 2
pt_sample_res = 64        # point_sampling

expt_id = 1

OVERFIT = args.of
overfit_idx = args.of_idx

batch_size = 10
if OVERFIT:
    batch_size = 1

ds_start = 0
ds_end = 508

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

torch.manual_seed(319)
np.random.seed(319)

train_new_ids_to_objs_path = f'data/{cat_name}_train_new_ids_to_objs_8_{ds_start}_{ds_end}.json'
with open(train_new_ids_to_objs_path, 'r') as f:
    train_new_ids_to_objs: Dict = json.load(f)
model_idx_to_anno_id = {}
for model_idx, anno_id in enumerate(train_new_ids_to_objs.keys()):
    model_idx_to_anno_id[model_idx] = anno_id

train_data_path = f'data/{cat_name}_train_{pt_sample_res}_8_{ds_start}_{ds_end}.hdf5'
train_data = h5py.File(train_data_path, 'r')
if OVERFIT:
    normalized_points = train_data['normalized_points'][overfit_idx:overfit_idx+1]
    values = train_data['values'][overfit_idx:overfit_idx+1]
else:
    normalized_points = train_data['normalized_points']
    values = train_data['values']

partnet_dataset = PartPtsDataset(
    root=f'data/{cat_name}_train_{pt_sample_res}_8_{ds_start}_{ds_end}_whole').to(device)
if OVERFIT:
    partnet_dataset = partnet_dataset[overfit_idx:overfit_idx+1]
partnet_loader = DataLoader(partnet_dataset, batch_size=batch_size)

num_points = normalized_points.shape[2]

if not OVERFIT:
    logs_path = os.path.join('logs', f'vae/vae_{expt_id}-bs-{batch_size}',
                             f'{pt_sample_res}')
    ckpt_dir = os.path.join(logs_path, 'ckpt')
    results_dir = os.path.join('results', f'vae/vae_{expt_id}-bs-{batch_size}',
                               f'{pt_sample_res}')
else:
    logs_path = os.path.join('logs',
                             f'vae/vae_{expt_id}-bs-{batch_size}-of-{overfit_idx}',
                             f'{pt_sample_res}')
    ckpt_dir = os.path.join(logs_path, 'ckpt')
    results_dir = os.path.join('results',
                               f'vae/vae_{expt_id}-bs-{batch_size}-of-{overfit_idx}',
                               f'{pt_sample_res}')
misc.check_dir(ckpt_dir)
misc.check_dir(results_dir)
print("results dir: ", results_dir)


# -------- model --------
model = VAE().to(device)
mse = torch.nn.MSELoss()
params = [p for _, p in model.named_parameters()]
optimizer = torch.optim.Adam(params, lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda x: max(0.0, 10**(-x*0.0002)))


def loss_f(pred_values, gt_values):
    recon_loss = mse(pred_values, gt_values)
    # kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss


def train_one_itr(partnet_data, batch_points, batch_values):
    occs = model.forward(partnet_data.pos, partnet_data.batch,
                         batch_points)
    loss = loss_f(occs, batch_values)

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
        for step, partnet_data in enumerate(partnet_loader):
            start = step * batch_size
            end = start + batch_size
            loss = train_one_itr(
                partnet_data,
                torch.from_numpy(normalized_points[start:end]).to(device, torch.float32),
                torch.from_numpy(values[start:end]).to(device, torch.float32))
            batch_loss += loss
        avg_batch_loss = batch_loss / batch_size
            
        show = 100 if OVERFIT else 10
        if (it) % show == 0 or it == (iterations - 1):
            info = f'Iteration {it} - loss: {avg_batch_loss:.8f}'
            print(info)

        save = 1000 if OVERFIT else 500
        if (it) % save == 0 or it == (iterations - 1):
            torch.save({
                'epoch': it,
                'model_state_dict': model.state_dict(),
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
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    results_dir = os.path.join(results_dir, anno_id)
    misc.check_dir(results_dir)

    unique_part_names, name_to_ori_ids_and_objs,\
        orig_obbs, entire_mesh, name_to_obbs, _, _, _ =\
        preprocess_data_8.merge_partnet_after_merging(anno_id)

    with open(f'data/{cat_name}_part_name_to_new_id_8_{ds_start}_{ds_end}.json', 'r') as f:
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
        partnet_dataset = partnet_dataset[model_idx:model_idx+1]
    partnet_loader = DataLoader(partnet_dataset, batch_size=1)

    with torch.no_grad():
        for partnet_data in partnet_loader:
            pred_values = model.forward(partnet_data.pos,
                                        partnet_data.batch,
                                        query_points)

    mag = 0.6

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
                            mag=mag, white_bg=False)
    
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

    obbs_of_interest = part_obbs
    import itertools
    obbs_of_interest = list(itertools.chain(*obbs_of_interest))

    visualize.save_obbs_vis(obbs_of_interest,
                            obbs_path, mag=0.6, white_bg=False)
    

    visualize.stitch_imges(
        os.path.join(results_dir,f'{anno_id}_results.png'),
        image_paths=lst_paths,
        adj=200)

    exit(0)

