import os
import json
import torch
import kaolin
from typing import Dict
from flexi.flexicubes import FlexiCubes
from flexi import render, util

train_new_ids_to_objs_path = f'data/Chair_train_new_ids_to_objs_17_0_100.json'
with open(train_new_ids_to_objs_path, 'r') as f:
    train_new_ids_to_objs: Dict = json.load(f)
model_idx_to_anno_id = {}
for model_idx, anno_id in enumerate(train_new_ids_to_objs.keys()):
    model_idx_to_anno_id[model_idx] = anno_id

partnet_dir = '/datasets/PartNet'

timelapse_dir = os.path.join('results/flexi', 'training_timelapse')
print("timelapse dir: ", timelapse_dir)
timelapse = kaolin.visualize.Timelapse(timelapse_dir)

device = 'cuda'
lr = 0.01
iterations = 1000
train_res = [512, 512]
fc_voxel_grid_res = 31
model_idx = 2

anno_id = model_idx_to_anno_id[model_idx]
obj_dir = os.path.join(partnet_dir, anno_id, 'vox_models')
gt_mesh_path = os.path.join(obj_dir, f'{anno_id}.obj')
gt_mesh = util.load_mesh(gt_mesh_path, device)

timelapse.add_mesh_batch(category='gt_mesh',
                         vertices_list=[gt_mesh.vertices.cpu()],
                         faces_list=[gt_mesh.faces.cpu()])

fc = FlexiCubes(device)
x_nx3, cube_fx8 = fc.construct_voxel_grid(fc_voxel_grid_res)
x_nx3 = 2*x_nx3

sdf = torch.rand_like(x_nx3[:,0]) - 0.1 # randomly init SDF
sdf    = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
deform = torch.nn.Parameter(torch.zeros_like(x_nx3), requires_grad=True)

def lr_schedule(iter):
    return max(0.0, 10**(-(iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.    

optimizer = torch.optim.Adam([sdf, deform], lr=lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x)) 

for it in range(iterations):
        optimizer.zero_grad()
        # sample random camera poses
        mv, mvp = render.get_random_camera_batch(
            8, iter_res=train_res, device=device, use_kaolin=False)
        # render gt mesh
        target = render.render_mesh_paper(gt_mesh, mv, mvp, train_res)
        
        # extract and render FlexiCubes mesh
        grid_verts = x_nx3 + (2-1e-8) / (fc_voxel_grid_res * 2) * torch.tanh(deform)
        
        vertices, faces, L_dev = fc(
             grid_verts, sdf, cube_fx8, fc_voxel_grid_res, training=True)

        flexicubes_mesh = util.Mesh(vertices, faces)
        buffers = render.render_mesh_paper(flexicubes_mesh, mv, mvp, train_res)
        
        # evaluate reconstruction loss
        mask_loss = (buffers['mask'] - target['mask']).abs().mean()
        depth_loss = (((((buffers['depth'] - (target['depth']))* target['mask'])**2).sum(-1)+1e-8)).sqrt().mean() * 10

        total_loss = mask_loss + depth_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if (it) % 100 == 0 or it == (iterations - 1): 
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