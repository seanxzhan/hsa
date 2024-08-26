import os
import json
import torch
import kaolin
import trimesh
from utils import misc, ops


name_to_cat = {
    'Chair': '03001627',
    'Lamp': '03636649',
    'Table': '04379243',
    'Earphone': '03261776'
}
cat_name = 'Chair'
cat_id = name_to_cat[cat_name]

pt_sample_res = 64

partnet_dir = '/datasets/PartNet'
partnet_index_path = '/sota/partnet_dataset/stats/all_valid_anno_info.txt'
stats_dir = '/sota/partnet_dataset/stats'
index_path = stats_dir + '/all_valid_anno_info.txt'
train_models_path = stats_dir + f'/train_val_test_split/{cat_name}.train.json'
val_models_path = stats_dir + f'/train_val_test_split/{cat_name}.val.json'
test_models_path = stats_dir + f'/train_val_test_split/{cat_name}.test.json'
after_merge_label_ids = stats_dir + f'/after_merging_label_ids/{cat_name}.txt'
OBB_REP_SIZE = 15

train_f = open(train_models_path); val_f = open(val_models_path); test_f = open(test_models_path)
train_ids = json.load(train_f); val_ids = json.load(val_f); test_ids = json.load(test_f)

results_dir = 'final/sdfusion'

def save_mesh_render_and_mask(anno_id):
    from utils import visualize
    partnet_objs_dir = os.path.join(partnet_dir, anno_id, 'objs')
    obj_paths = [os.path.join(partnet_objs_dir, x)
                 for x in os.listdir(partnet_objs_dir)]
    mesh_parts = [trimesh.load_mesh(x) for x in obj_paths]
    mesh: trimesh.Trimesh = trimesh.util.concatenate(mesh_parts)
    mesh_verts = torch.from_numpy(mesh.vertices).to(torch.float32).unsqueeze(0)
    mesh_verts = kaolin.ops.pointcloud.center_points(mesh_verts,
                                                     normalize=True)
    mesh_verts = mesh_verts.numpy()[0]
    normalized_mesh = trimesh.Trimesh(mesh_verts, mesh.faces)
    normalized_mesh.visual.vertex_colors = [31, 119, 180]
    out_dir = os.path.join(results_dir, anno_id); misc.check_dir(out_dir)
    render_path = os.path.join(out_dir, f'{anno_id}.png')
    mask_path = os.path.join(out_dir, f'{anno_id}-mask.png')
    visualize.save_mesh_vis(normalized_mesh,
                            render_path,
                            mag=0.8,
                            white_bg=True,
                            save_img=True)
    visualize.save_mesh_vis(normalized_mesh,
                            mask_path,
                            mag=0.8,
                            white_bg=True,
                            save_img=True,
                            seg=True)
    

def save_sdfusion_obj_renders(anno_id):
    from utils import visualize
    sdfusion_results_dir = '/sota/SDFusion/results'
    obj_path = os.path.join(sdfusion_results_dir, f'{anno_id}_out.obj')
    mesh: trimesh.Trimesh = trimesh.load_mesh(obj_path)
    trimesh.repair.fix_normals(mesh)
    norm_mesh = ops.export_mesh_norm(torch.from_numpy(mesh.vertices), 
                                     torch.from_numpy(mesh.faces))
    out_dir = os.path.join(results_dir, anno_id)
    gt_path = os.path.join(out_dir, f'{anno_id}.png')
    render_path = os.path.join(out_dir, f'{anno_id}-sdfusion.png')
    visualize.save_mesh_vis(norm_mesh,
                            render_path,
                            mag=0.8,
                            white_bg=True,
                            save_img=True)
    stitched_path = os.path.join(out_dir, f'{anno_id}-vs.png')
    img_paths = [gt_path, render_path]
    visualize.stitch_imges(stitched_path,
                           image_paths=img_paths)


if __name__ == "__main__":
    # for i in range(1):
    #     anno_id = test_ids[i]['anno_id']
    #     print(anno_id)
    #     save_mesh_render_and_mask(anno_id)

    for i in range(10, 50):
        anno_id = test_ids[i]['anno_id']
        save_sdfusion_obj_renders(anno_id)