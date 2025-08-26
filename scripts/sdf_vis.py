from utils import ops

def get_info_from_voxels(anno_id, res, entire_mesh):
    """Given ShapeNet model_id and pt_sample_res,
    return model voxels and binvox transformation
    """
    obj_dir = os.path.join(partnet_dir, anno_id, 'vox_models')
    misc.check_dir(obj_dir)
    gt_mesh_path = os.path.join(obj_dir, f'{anno_id}.obj')
    entire_mesh.export(gt_mesh_path)
    vox_path = os.path.join(obj_dir, f'{anno_id}_{res}.binvox')
    vox_c_path = os.path.join(obj_dir, f'{anno_id}_{res}_c.binvox')
    if not os.path.exists(vox_c_path):
        ops.setup_vox(obj_dir)
        ops.voxelize_obj(
            obj_dir,
            f'{anno_id}.obj',
            res,
            vox_c_path,
            vox_path)
        ops.teardown_vox(obj_dir)
    voxels = ops.load_voxels(vox_c_path)
    binvox_xform = transform.get_transform_from_binvox_centered(
        vox_c_path, vox_path)
    return voxels, binvox_xform     


mesh_path = "/projects/occflexi/results/deformed_meshes/paper/chair/objs/1339_before.obj"

