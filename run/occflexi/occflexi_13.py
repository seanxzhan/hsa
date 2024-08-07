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
from occ_networks.occflexi_network_9 import SDFDecoder, get_embedder
from utils import misc, ops, reconstruct, polyvis
from data_prep import preprocess_data_19

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str)
parser.add_argument('--train', action="store_true")
parser.add_argument('--it', type=int)
parser.add_argument('--test', action="store_true")
parser.add_argument('--test_idx', '--ti', type=int)
parser.add_argument('--mask', action="store_true")
parser.add_argument('--parts', '--p', nargs='+')
parser.add_argument('--recon_one_part', '--ro', action="store_true")
parser.add_argument('--recon_the_rest', '--rt', action="store_true")
args = parser.parse_args()

# ------------ hyper params ------------
name_to_cat = {
    'Chair': '03001627',
}
cat_name = 'Chair'
pt_sample_res = 64
occ_res = int(pt_sample_res / 2)
ds_start, ds_end = 0, 508
device = 'cuda'
lr = 0.001
iterations = 10000; iterations += 1
train_res = [512, 512]
fc_res = 31
num_shapes = 496
batch_size = 8
each_part_feat = 32
embed_dim = 128
dataset_id = 19
expt_id = 13
anchor_idx = -1
num_batches = num_shapes // batch_size

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

# ------------ logging ------------
results_dir = os.path.join('results', 'occflexi', f'occflexi_{expt_id}')
timelapse_dir = os.path.join(results_dir, 'training_timelapse')
print("timelapse dir: ", timelapse_dir)
timelapse = kaolin.visualize.Timelapse(timelapse_dir)
logs_dir = os.path.join('logs', 'occflexi', f'occflexi_{expt_id}')
print("logs dir: ", logs_dir)
ckpt_dir = os.path.join(logs_dir, 'ckpt'); misc.check_dir(ckpt_dir)
writer = SummaryWriter(os.path.join(logs_dir, 'summary'))

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

# ------------ part disentanglement info ------------
connectivity = [[0, 1], [0, 2], [1, 2], [2, 3]]
connectivity = torch.tensor(connectivity, dtype=torch.long).to(device)
num_union_nodes = adj.shape[1]
num_points = normalized_points.shape[1]
_, num_parts = part_num_indices.shape
all_fg_part_indices = []
for i in range(num_shapes):
    indices = preprocess_data_19.convert_flat_list_to_fg_part_indices(
        part_num_indices[i], all_indices[i])
    all_fg_part_indices.append(np.array(indices, dtype=object))

# ------------ init flexicubes ------------
fc = FlexiCubes(device)
x_nx3, cube_fx8 = fc.construct_voxel_grid(fc_res)
x_nx3 *= 1.15
x_nx3 = x_nx3.to(device)
# flexi_verts: torch.Tensor = x_nx3.to(device).unsqueeze(0).expand(batch_size, -1, -1)
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

# ------------ init gt meshes ------------
if args.train:
    print("loading gt meshes")
    gt_meshes = [my_load_mesh(s) for s in tqdm(range(0, num_shapes))]
    timelapse.add_mesh_batch(category='gt_mesh',
                             vertices_list=[gt_meshes[anchor_idx].vertices.cpu()],
                             faces_list=[gt_meshes[anchor_idx].faces.cpu()])
    occ_pts = torch.argwhere(torch.from_numpy(
        occ[np.arange(0, num_shapes)[anchor_idx]]).reshape([fc_res+1]*3) == 1.0)
    occ_pts = occ_pts/(fc_res+1) - 0.5
    timelapse.add_pointcloud_batch(category='gt_occ', pointcloud_list=[occ_pts])

# ------------ embeddings ------------
occ_embeddings = torch.nn.Embedding(num_shapes,
                                    num_parts*each_part_feat).to(device)
torch.nn.init.normal_(occ_embeddings.weight.data, 0.0,
                      1 / math.sqrt(num_parts*each_part_feat))

# ------------ network ------------
occ_model = SDFDecoder(num_parts=num_parts,
                       feature_dims=each_part_feat,
                       internal_dims=128,
                       hidden=4,
                       multires=2).to(device)
occ_model_params = [p for _, p in occ_model.named_parameters()]
embed_fn, _ = get_embedder(2)

# ------------ optimizer ------------
optimizer = torch.optim.Adam([{"params": occ_model_params, "lr": lr},
                              {"params": occ_embeddings.parameters(), "lr": lr}])
# def lr_schedule(iter):
#     return max(0.0, 10**(-(iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.    
# def lr_schedule(iter):
#     return max(0.0, 10**(-(iter)*0.0001)) # Exponential falloff from [1.0, 0.1] over 10k epochs.    
# scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer, lr_lambda=lambda x: lr_schedule(x))

# ------------ loss ------------
mse = torch.nn.MSELoss()
def loss_f(pred_values, gt_values):
    recon_loss = mse(pred_values, gt_values)
    return recon_loss
# def mesh_loss_schedule(iter):
#     # approaches 1 from 0 for 5000 iterations
#     return -10**(-0.0002*iter) + 1.1
# def occ_loss_schedule(iter):
#     # approaches 0.1 from 1 for 10000 iterations
#     return max(0, -10**(-0.0001*iter))

# ------------ batch loading ------------
def load_batch(batch_idx, batch_size, start=None, end=None):
    if start is None:
        start = batch_idx*batch_size
    if end is None:
        end = start + batch_size
    return all_fg_part_indices[start:end],\
        torch.from_numpy(normalized_points[start:end]).to(device, torch.float32),\
        torch.from_numpy(values[start:end]).to(device, torch.float32),\
        occ_embeddings(torch.arange(start, end).to(device)),\
        torch.from_numpy(node_features[start:end, :, :3]).to(device, torch.float32),\
        torch.from_numpy(adj[start:end]).to(device, torch.long),\
        torch.from_numpy(part_nodes[start:end]).to(device, torch.long),\
        torch.from_numpy(xforms[start:end, :, :3, 3]).to(device, torch.float32),\
        torch.from_numpy(relations[start:end, :, :3, 3]).to(device, torch.float32)
def load_meshes(batch_idx, batch_size):
    start = batch_idx*batch_size
    end = start + batch_size
    return [gt_meshes[i] for i in range(start, end)]

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
    return mask_loss+depth_loss, vertices, faces

# ------------ occupancy ------------
def run_occ(batch_size, masked_indices, batch_points, batch_embed,
            batch_node_feat, batch_adj, batch_part_nodes, mask_flexi=False):
    # ------------ parts, points, occ masks ------------
    parts_mask = torch.zeros((batch_embed.shape[0], num_parts)).to(device, torch.float32)
    parts_mask[:, masked_indices] = 1
    parts_mask = torch.repeat_interleave(parts_mask, each_part_feat, dim=-1)
    points_mask = torch.zeros((batch_size, num_parts, 1, 1)).to(device, torch.float32)
    points_mask[:, masked_indices] = 1
    occ_mask = torch.zeros((batch_size, 1, num_parts)).to(device, torch.float32)
    occ_mask[:, :, masked_indices] = 1

    # ------------ learning bbox & xforms ------------
    batch_vec = torch.arange(start=0, end=batch_size).to(device)
    batch_vec = torch.repeat_interleave(batch_vec, num_union_nodes)
    batch_mask = torch.sum(batch_part_nodes, dim=1)
    learned_geom, learned_xforms = occ_model.learn_geom_xform(batch_node_feat,
                                                              batch_adj,
                                                              batch_mask,
                                                              batch_vec)
    pairwise_xforms = learned_xforms[:, connectivity]
    learned_relations = pairwise_xforms[:, :, 1] - pairwise_xforms[:, :, 0]

    # ------------ occ ------------
    transformed_points =\
        batch_points.unsqueeze(1).expand(-1, num_parts, -1, -1) +\
        learned_xforms.unsqueeze(2)

    occs1 = occ_model.forward(
        transformed_points.masked_fill(points_mask==1,torch.tensor(0)),
        batch_embed.masked_fill(parts_mask==1, torch.tensor(0)))
    occs1 = occs1.masked_fill(occ_mask==1,torch.tensor(float('-inf')))
    pred_values1, _ = torch.max(occs1, dim=-1, keepdim=True)

    pred_values2, _ = torch.max(occ_model.forward(
        transformed_points, batch_embed), dim=-1, keepdim=True)

    # ------------ flexi ------------
    transformed_flexi_verts =\
        x_nx3.unsqueeze(0).unsqueeze(1).expand(batch_size, num_parts, -1, -1) +\
        learned_xforms.unsqueeze(2)

    if not mask_flexi:
        pred_verts_occ, _ = torch.max(occ_model.forward(
            transformed_flexi_verts, batch_embed), dim=-1, keepdim=True)
    else:
        pred_verts_occ, _ = torch.max(
            occ_model.forward(
                transformed_flexi_verts.masked_fill(points_mask==1,torch.tensor(0)),
                batch_embed.masked_fill(parts_mask==1, torch.tensor(0))
                ).masked_fill(occ_mask==1,torch.tensor(float('-inf'))),
            dim=-1, keepdim=True)

    comp_sdf = ops.bin2sdf_torch_3( 
        pred_verts_occ.view(-1, fc_res+1, fc_res+1, fc_res+1))
    
    return learned_xforms, learned_geom, learned_relations,\
           pred_values1, pred_values2, pred_verts_occ, comp_sdf

all_indices = torch.arange(0, num_parts).to(device, torch.long)

# ------------ training ------------
if args.train:
    for it in range(iterations):
        itr_loss = 0
        for b in range(num_batches):
            optimizer.zero_grad()

            # ------------ loading batch ------------        
            batch_fg_part_indices, batch_points, batch_values,\
                batch_embed, batch_node_feat, batch_adj, batch_part_nodes,\
                batch_xforms, batch_relations =\
                load_batch(b, batch_size)
            batch_gt_meshes = load_meshes(b, batch_size)

            # ------------ random masking ------------
            n = (it * num_batches + b) % num_parts
            masked_indices = all_indices[list(set(range(0, num_parts))-set([n]))]

            # ------------ gt value mask ------------
            val_mask = torch.ones_like(batch_values).to(device, torch.float32)
            for i in range(batch_size):
                fg_part_indices = batch_fg_part_indices[i]
                fg_part_indices_masked = fg_part_indices[masked_indices.cpu().numpy()]
                if len(fg_part_indices_masked) != 0:
                    fg_part_indices_masked = np.concatenate(fg_part_indices_masked, axis=0)
                else:
                    fg_part_indices_masked = np.array([])
                val_mask[i][fg_part_indices_masked] = 0
            modified_values = batch_values * val_mask

            # ------------ gt bbox geom ------------
            batch_geom = torch.einsum('ijk, ikm -> ijm',
                                      batch_part_nodes.to(torch.float32),
                                      batch_node_feat)

            # ------------ occflexi prediction ------------
            learned_xforms, learned_geom, learned_relations,\
            pred_values1, pred_values2, pred_verts_occ, comp_sdf =\
                run_occ(batch_size, masked_indices, batch_points, batch_embed,
                        batch_node_feat, batch_adj, batch_part_nodes)

            # ------------ flexi loss ------------
            mesh_loss = 0
            for s in range(batch_size):
                one_mesh_loss, vertices, faces = run_flexi(
                    torch.flatten(comp_sdf[s]).unsqueeze(0), batch_gt_meshes[s],
                    pred_verts_occ[s])
                mesh_loss += one_mesh_loss
            mesh_loss /= batch_size

            # ------------ occ loss ------------
            loss1 = loss_f(pred_values1, modified_values)
            loss2 = loss_f(pred_values2, batch_values)
            loss_bbox_geom = loss_f(
                embed_fn(learned_geom.view(-1, 3)),
                embed_fn(batch_geom.view(-1, 3)),)
            loss_xform = loss_f(
                embed_fn(learned_xforms.view(-1, 3)),
                embed_fn(batch_xforms.view(-1, 3)))
            loss_relations = loss_f(
                embed_fn(learned_relations.view(-1, 3)),
                embed_fn(batch_relations.view(-1, 3)))
            total_loss = loss1 + loss2 + mesh_loss + loss_bbox_geom +\
                         10 * loss_xform + 10 * loss_relations

            total_loss.backward()
            optimizer.step()
            # scheduler.step()
            itr_loss += total_loss.detach().cpu().numpy()
        itr_loss /= num_shapes

        # ------------ logging ------------
        writer.add_scalar('iteration loss', total_loss, it)
        writer.add_scalar('loss1', loss1, it)
        writer.add_scalar('loss2', loss2, it)
        writer.add_scalar('mesh', mesh_loss, it)
        writer.add_scalar('bbox', loss_bbox_geom, it)
        writer.add_scalar('xform', loss_xform, it)
        writer.add_scalar('relations', loss_relations, it)

        # ------------ print loss & saves occflexi ------------
        if (it) % 50 == 0 or it == (iterations - 1): 
            with torch.no_grad():
                vertices, faces, L_dev = fc(
                    x_nx3, torch.flatten(comp_sdf[-1]), cube_fx8,
                    fc_res, training=False)
            print('Iteration {} - loss: {:.5f}, '.format(it, total_loss)+
                  '# of mesh vertices: {}, # of mesh faces: {}'.format(
                    vertices.shape[0], faces.shape[0]))
            # print('Iteration {} - loss: {:.5f}, '.format(it, total_loss))
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
        
        # ------------ checkpointing ------------
        if (it) % 250 == 0 or it == (iterations - 1): 
            torch.save({
                'epoch': it,
                'model_state_dict': occ_model.state_dict(),
                'embeddings_state_dict': occ_embeddings.state_dict(),
                'total_loss': total_loss,
                }, os.path.join(ckpt_dir, f'model_{it}.pt'))
    writer.close()

# ------------ reconstruction ------------
if args.test:
    from utils import visualize
    white_bg = True
    it = args.it
    model_idx = args.test_idx
    anno_id = model_idx_to_anno_id[model_idx]
    model_id = misc.anno_id_to_model_id(partnet_index_path)[anno_id]
    print(f"anno id: {anno_id}, model id: {model_id}")
    results_dir = os.path.join(results_dir, 'recon', anno_id)
    misc.check_dir(results_dir)
    print("results dir: ", results_dir)

    # ------------ loading model, embedding, and data ------------
    checkpoint = torch.load(os.path.join(ckpt_dir, f'model_{it}.pt'))
    occ_model.load_state_dict(checkpoint['model_state_dict'])
    occ_embeddings = torch.nn.Embedding(num_shapes, num_parts*each_part_feat).to(device)
    occ_embeddings.load_state_dict(checkpoint['embeddings_state_dict'])
    _, _, _, batch_embed, batch_node_feat, batch_adj, batch_part_nodes, _, _ =\
        load_batch(0, 0, model_idx, model_idx+1)
    batch_gt_meshes = [my_load_mesh(model_idx)]

    # ------------ gt bboxes ------------
    unique_part_names, name_to_ori_ids_and_objs,\
        orig_obbs, entire_mesh, name_to_obbs, _, _, _, _ =\
        preprocess_data_19.merge_partnet_after_merging(anno_id)
    with open(f'data/{cat_name}_part_name_to_new_id_19_{ds_start}_{ds_end}.json', 'r') as f:
        unique_name_to_new_id = json.load(f)
    part_obbs = []
    unique_names = list(unique_name_to_new_id.keys())
    model_part_names = list(name_to_obbs.keys())
    for i, un in enumerate(unique_names):
        if not un in model_part_names:
            part_obbs.append([])
            continue
        part_obbs.append([name_to_obbs[un]])

    # ------------ img paths ------------
    gt_color = [31, 119, 180, 255]
    mesh_pred_path = os.path.join(results_dir, 'mesh_occ.png')
    mesh_gt_path = os.path.join(results_dir, 'mesh_gt.png')
    learned_obbs_path = os.path.join(results_dir, 'obbs_pred.png')
    obbs_path = os.path.join(results_dir, 'obbs_gt.png')
    mesh_flexi_path = os.path.join(results_dir, 'mesh_flexi.png')
    lst_paths = [
        obbs_path, mesh_gt_path, learned_obbs_path, mesh_pred_path, mesh_flexi_path]

    # ------------ making query points ------------
    query_points = reconstruct.make_query_points(pt_sample_res)
    query_points = torch.from_numpy(query_points).to(device, torch.float32)
    query_points = query_points.unsqueeze(0)
    bs, num_points, _ = query_points.shape

    with torch.no_grad():
        # ------------ dealing with masking indices ------------
        if args.mask:
            parts = args.parts
            parts = [int(x) for x in parts]
            if args.recon_one_part:
                # only reconstruct part specified by parts
                masked_indices = list(set(range(num_parts)) - set(parts))
                masked_indices = torch.Tensor(masked_indices).to(device, torch.long)
            if args.recon_the_rest:
                # don't reconstruct stuff in masked_indices
                masked_indices = torch.Tensor(parts).to(device, torch.long)
            print(f"masking parts: {masked_indices.cpu().numpy().tolist()}")
        else:
            masked_indices = torch.Tensor([]).to(device, torch.long)
            print("reconstructing...")

        learned_xforms, learned_geom, learned_relations,\
            pred_values1, pred_values2, pred_verts_occ, comp_sdf =\
            run_occ(1, masked_indices, query_points, batch_embed,
                    batch_node_feat, batch_adj, batch_part_nodes, mask_flexi=True)

        one_mesh_loss, flexi_vertices, flexi_faces =\
            run_flexi(torch.flatten(comp_sdf[0]).unsqueeze(0),
                      batch_gt_meshes[0], pred_verts_occ[0])

    # ------------ pred bboxes ------------
    # NOTE: this is using gt bbox geom atm
    learned_xforms = learned_xforms[0].cpu().numpy()
    learned_obbs_of_interest = [[]] * num_parts
    for i in range(num_parts):
        ext = extents[model_idx, i]
        learned_xform = np.eye(4)
        learned_xform[:3, 3] = -learned_xforms[i]
        ext_xform = (ext, learned_xform)
        learned_obbs_of_interest[i] = [ext_xform]

    mag = 0.8

    # ------------ mesh from occupancy ------------
    print("exporting occ mesh")
    sdf_grid = torch.reshape(
        pred_values1,
        (1, pt_sample_res, pt_sample_res, pt_sample_res))
    sdf_grid = torch.permute(sdf_grid, (0, 2, 1, 3))
    occ_vertices, occ_faces =\
        kaolin.ops.conversions.voxelgrids_to_trianglemeshes(sdf_grid)
    mesh_occ = ops.export_mesh_norm(occ_vertices[0].cpu(), occ_faces[0].cpu(),
                                    os.path.join(results_dir, 'mesh_occ.obj'))
    visualize.save_mesh_vis(mesh_occ, mesh_pred_path,
                            mag=mag, white_bg=white_bg)
    
    # ------------ mesh from flexicubes ------------
    print("exporting flexi mesh")
    flexi_vertices = flexi_vertices.detach().cpu()
    flexi_faces = flexi_faces.detach().cpu()
    mesh_flexi = ops.export_mesh_norm(flexi_vertices, flexi_faces,
                                      os.path.join(results_dir, 'mesh_flexi.obj'))
    flexi_mesh_vis = visualize.save_mesh_vis(mesh_flexi, mesh_flexi_path,
                                             mag=mag, white_bg=white_bg)
    print("saving flexi sdf")
    np.save(os.path.join(results_dir, f'flexi_sdf.npy'),
            comp_sdf[0].detach().cpu().numpy())

    masked_indices = masked_indices.cpu().numpy().tolist()
    unmasked_indices = list(set(range(num_parts)) - set(masked_indices))

    # ------------ gt mesh ------------
    if not args.mask:
        obj_dir = os.path.join(partnet_dir, anno_id, 'vox_models')
        assert os.path.exists(obj_dir)
        gt_mesh_path = os.path.join(obj_dir, f'{anno_id}.obj')
        gt_mesh = trimesh.load(gt_mesh_path, file_type='obj', force='mesh')
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
        if args.recon_one_part:
            # only show stuff in masked_indices
            gt_mesh = trimesh.util.concatenate(
                [lst_of_part_meshes[x] for x in parts])
        if args.recon_the_rest:
            # don't show stuff in masked_indices
            gt_mesh = trimesh.util.concatenate(
                [lst_of_part_meshes[x] for x in unmasked_indices])
    mesh_gt = ops.export_mesh_norm(torch.from_numpy(gt_mesh.vertices),
                                   torch.from_numpy(gt_mesh.faces),
                                   os.path.join(results_dir, 'mesh_gt.obj'))
    mesh_gt.visual.vertex_colors = gt_color
    visualize.save_mesh_vis(mesh_gt, mesh_gt_path,
                            mag=mag, white_bg=white_bg)

    # ------------ evaluation ------------
    gt_samples, _ = trimesh.sample.sample_surface(mesh_gt, 10000, seed=319)
    gt_samples = torch.from_numpy(gt_samples).to(device)
    occ_samples, _ = trimesh.sample.sample_surface(mesh_occ, 10000, seed=319)
    occ_samples = torch.from_numpy(occ_samples).to(device)
    flexi_samples, _ = trimesh.sample.sample_surface(mesh_flexi, 10000, seed=319)
    flexi_samples = torch.from_numpy(flexi_samples).to(device)
    occ_chamfer = kaolin.metrics.pointcloud.chamfer_distance(
        occ_samples.unsqueeze(0), gt_samples.unsqueeze(0)).mean()
    flexi_chamfer = kaolin.metrics.pointcloud.chamfer_distance(
        flexi_samples.unsqueeze(0), gt_samples.unsqueeze(0)).mean()
    print("chamfer distance, gt-occ: ", occ_chamfer.cpu().numpy())
    print("chamfer distance, gt-flexi: ", flexi_chamfer.cpu().numpy())

    # ------------ obbs visualization ------------
    print("exporting obbs")
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
    
    # ------------ stitching results ------------
    print("exporting stitched results")
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