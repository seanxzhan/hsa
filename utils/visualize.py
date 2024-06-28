import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import trimesh
import numpy as np
import pyrender
import colorsys
from PIL import Image
from scipy.spatial.transform import Rotation as R
from pyrender.constants import RenderFlags

LIMITS = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]

def hex2rgb(h):
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

def increase_saturation(rgb, percent):
    # convert RGB values to HSV (hue, saturation, value) format
    hsv = colorsys.rgb_to_hsv(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    # increase the saturation by a percentage
    hsv = (hsv[0], hsv[1] + percent / 100.0, hsv[2])
    # convert back to RGB format
    rgb = tuple(max(0, min(int(x * 255), 255)) for x in colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2]))
    return rgb

def hex2rgb_sat(h, percent):
    color = hex2rgb(h)
    return increase_saturation(color, percent)

def get_tab_20():
    return [
        hex2rgb('17becf'),
        hex2rgb('dbdb8d'),
        hex2rgb('bcbd22'),
        hex2rgb('c7c7c7'),
        hex2rgb('7f7f7f'),
        hex2rgb('f7b6d2'),
        hex2rgb('e377c2'),
        hex2rgb('c49c94'),
        hex2rgb('8c564b'),
        hex2rgb('c5b0d5'),
        hex2rgb('9467bd'),
        hex2rgb('ff9896'),
        hex2rgb('d62728'),
        hex2rgb('98df8a'),
        hex2rgb('2ca02c'),
        hex2rgb('ffbb78'),
        hex2rgb('ff7f0e'),
        hex2rgb('aec7e8'),
        hex2rgb('1f77b4'),
        hex2rgb('9edae5'),
    ] * 3

def get_tab_20_saturated(percent):
    return [
        hex2rgb_sat('17becf', percent),
        hex2rgb_sat('dbdb8d', percent),
        hex2rgb_sat('bcbd22', percent),
        hex2rgb_sat('c7c7c7', percent),
        hex2rgb_sat('7f7f7f', percent),
        hex2rgb_sat('f7b6d2', percent),
        hex2rgb_sat('e377c2', percent),
        hex2rgb_sat('c49c94', percent),
        hex2rgb_sat('8c564b', percent),
        hex2rgb_sat('c5b0d5', percent),
        hex2rgb_sat('9467bd', percent),
        hex2rgb_sat('ff9896', percent),
        hex2rgb_sat('d62728', percent),
        hex2rgb_sat('98df8a', percent),
        hex2rgb_sat('2ca02c', percent),
        hex2rgb_sat('ffbb78', percent),
        hex2rgb_sat('ff7f0e', percent),
        hex2rgb_sat('aec7e8', percent),
        hex2rgb_sat('1f77b4', percent),
        hex2rgb_sat('9edae5', percent),
    ] * 3

def save_fig(plt, title, img_path, rotate=False, transparent=False):
    plt.title(title)
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0, dpi=300, 
                transparent=transparent)
    if rotate:
        im = Image.open(img_path)
        im = im.rotate(90)
        im.save(img_path)

def save_mesh_vis(trimesh_obj, out_path, mesh_y_rot=-45, mag=1,
                  white_bg=False, save_img=True):
    mesh = pyrender.Mesh.from_trimesh(trimesh_obj, smooth=False)

    if white_bg:
        scene = pyrender.Scene(bg_color=[256,256,256,256],
                           ambient_light=[0.3,0.3,0.3,1.0])
    else:
        scene = pyrender.Scene(bg_color=[0,0,0,0],
                            ambient_light=[0.3,0.3,0.3,1.0])
    
    rotation_mat_y = np.identity(4)
    rot = R.from_euler('y', mesh_y_rot, degrees=True).as_matrix()
    rotation_mat_y[:3, :3] = rot
    scene.add(mesh, pose=rotation_mat_y)

    # add camera
    mag = mag
    cam = pyrender.OrthographicCamera(xmag=mag, ymag=mag)
    translation_mat = np.array([
        [1.2, 0, 0, 0.15],
        [0, 1.2, 0, 0],
        [0, 0, 1.2, 2],
        [0, 0, 0, 1]
    ])

    rotation_mat_x = np.identity(4)
    rot = R.from_euler('x', -30, degrees=True).as_matrix()
    # rot = R.from_euler('y', 90, degrees=True).as_matrix()
    rotation_mat_x[:3, :3] = rot
    cam_pose = rotation_mat_x @ translation_mat
    scene.add(cam, pose=cam_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    light_mat = np.identity(4)
    rot = R.from_euler('x', -90, degrees=True).as_matrix()
    light_mat[:3, :3] = rot
    scene.add(light, pose=light_mat)

    flags = RenderFlags.RGBA
    r = pyrender.OffscreenRenderer(viewport_width=500,
                                   viewport_height=500,
                                   point_size=1.0)
    if white_bg:
        color, _ = r.render(scene)
    else:
        color, _ = r.render(scene, flags=flags)
    r.delete()

    im = Image.fromarray(color)
    if save_img:
        im.save(out_path)
    return im


def save_obbs_vis(obbs, out_path, mesh_y_rot=-45, mag=1,
                  white_bg=False, save_img=True, show_coord=False,
                  name_to_obbs=None, unmasked_indices=None):
    """obbs: a list of tuples x, where x = (extents, transform)
    """
    if name_to_obbs is None:    
        # obb_meshes = []
        # for x in obbs:
        #     obb_mesh = trimesh.creation.box(extents=x[0], transform=x[1])
        #     # obb_mesh.visual.face_colors = [100, 100, 100, 255]
        #     obb_meshes.append(obb_mesh)

        # material = pyrender.MetallicRoughnessMaterial(
        #     baseColorFactor=[0.12, 0.46, 0.70, 0.4],  alphaMode='BLEND')
        # obb_meshes = [pyrender.Mesh.from_trimesh(x, smooth=False, material=material)
        #             for x in obb_meshes]

        obb_meshes = []
        for i, obb in enumerate(obbs):
            if i == 2:
                base_color = [148.0/255, 201.0/255, 107.0/255, 0.9]
            if i == 3:
                base_color = [221.0/255, 137.0/255, 133.0/255, 0.9]
            if i == 1:
                base_color = [173.0/255, 157.0/255, 190.0/255, 0.9]
            if i == 0:
                base_color = [150.0/255, 118.0/255, 96.0/255, 0.9]
            if unmasked_indices is not None and len(unmasked_indices) == 1:
                if unmasked_indices == [2]:
                    base_color = [148.0/255, 201.0/255, 107.0/255, 0.9]
                if unmasked_indices == [3]:
                    base_color = [221.0/255, 137.0/255, 133.0/255, 0.9]
                if unmasked_indices == [1]:
                    base_color = [173.0/255, 157.0/255, 190.0/255, 0.9]
                if unmasked_indices == [0]:
                    base_color = [150.0/255, 118.0/255, 96.0/255, 0.9]
            # base_color = [148.0/255, 201.0/255, 107.0/255, 0.9]
            # base_color = [0.0] * 4
            # else:
            #     # base_color = [0.0, 0.0, 0.0, 0.0]
            #     continue
            obb_meshes.append(
                pyrender.Mesh.from_trimesh(
                    trimesh.creation.box(extents=obb[0], transform=obb[1]),
                    smooth=False,
                    material=pyrender.MetallicRoughnessMaterial(
                        baseColorFactor=base_color, alphaMode='BLEND')))
    else:
        obb_meshes = []
        for name, obb in name_to_obbs.items():
            if name == 'leg':
                base_color = [0.70, 0.46, 0.12, 0.4]
            else:
                base_color = [0.12, 0.46, 0.70, 0.4]
            obb_meshes.append(
                pyrender.Mesh.from_trimesh(
                    trimesh.creation.box(extents=obb[0], transform=obb[1]),
                    smooth=False,
                    material=pyrender.MetallicRoughnessMaterial(
                        baseColorFactor=base_color, alphaMode='BLEND')))

    rotation_mat_y = np.identity(4)
    rot = R.from_euler('y', mesh_y_rot, degrees=True).as_matrix()
    rotation_mat_y[:3, :3] = rot

    if white_bg:
        scene = pyrender.Scene(bg_color=[256,256,256,256],
                            ambient_light=[0.3,0.3,0.3,1.0])
    else:
        scene = pyrender.Scene(bg_color=[0,0,0,0],
                            ambient_light=[0.3,0.3,0.3,1.0])
    for x in obb_meshes: scene.add(x, pose=rotation_mat_y)

    edge_meshes = []
    for i, x in enumerate(obbs):
        # if i != 0:
        #     continue
        edges, lengths, edge_xforms = compute_obb_edges(x[0], x[1])
        for i in range(len(edges)):
            edge_mesh = trimesh.creation.cylinder(
                0.002, height=lengths[i], transform=edge_xforms[i])
            edge_meshes.append(edge_mesh)
    
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0, 0, 0])
    edge_meshes = [pyrender.Mesh.from_trimesh(x, smooth=False, material=material)
                  for x in edge_meshes]
    for x in edge_meshes: scene.add(x, pose=rotation_mat_y)

    if show_coord:
        coord_meshes = []
        for obb in obbs:

            rad = 0.01
            h = 0.1

            translation = trimesh.transformations.translation_matrix(obb[5])

            transx = trimesh.transformations.translation_matrix([h/2, 0, 0])
            transy = trimesh.transformations.translation_matrix([0, h/2, 0])
            transz = trimesh.transformations.translation_matrix([0, 0, h/2])

            rotx = trimesh.geometry.align_vectors([0,0,1], obb[2])
            roty = trimesh.geometry.align_vectors([0,0,1], obb[3])
            rotz = trimesh.geometry.align_vectors([0,0,1], obb[4])

            x_xform = np.dot(translation, rotx @ transz)
            y_xform = np.dot(translation, roty @ transz)
            z_xform = np.dot(translation, rotz @ transz)

            x_cylinder = trimesh.creation.cylinder(rad, h, transform=x_xform)
            x_cylinder.visual.vertex_colors = [255, 0, 0]

            y_cylinder = trimesh.creation.cylinder(rad, h, transform=y_xform)
            y_cylinder.visual.vertex_colors = [0, 255, 0]

            z_cylinder = trimesh.creation.cylinder(rad, h, transform=z_xform)
            z_cylinder.visual.vertex_colors = [0, 0, 255]

            coord_meshes.append(x_cylinder)
            coord_meshes.append(y_cylinder)
            coord_meshes.append(z_cylinder)

        coord_meshes = [pyrender.Mesh.from_trimesh(x, smooth=True)
                       for x in coord_meshes]
        for x in coord_meshes: scene.add(x, pose=rotation_mat_y)

    # add camera
    mag = mag
    cam = pyrender.OrthographicCamera(xmag=mag, ymag=mag)
    translation_mat = np.array([
        [1.2, 0, 0, 0.15],
        # [0, 1.2, 0, -0.13],
        [0, 1.2, 0, 0],
        [0, 0, 1.2, 2],
        [0, 0, 0, 1]
    ])

    rotation_mat_x = np.identity(4)
    rot = R.from_euler('x', -30, degrees=True).as_matrix()
    rotation_mat_x[:3, :3] = rot
    cam_pose = rotation_mat_x @ translation_mat
    scene.add(cam, pose=cam_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    light_mat = np.identity(4)
    rot = R.from_euler('x', -90, degrees=True).as_matrix()
    light_mat[:3, :3] = rot
    scene.add(light, pose=light_mat)

    flags = RenderFlags.RGBA
    r = pyrender.OffscreenRenderer(viewport_width=500,
                                   viewport_height=500,
                                   point_size=1.0)
    if white_bg:
        color, _ = r.render(scene)
    else:
        color, _ = r.render(scene, flags=flags)
    r.delete()

    im = Image.fromarray(color)
    if save_img:
        im.save(out_path)
    return im

def compute_obb_edges(extents, transform):
    # Define the 8 vertices in local coordinates
    ex, ey, ez = extents / 2
    vertices_local = np.array([
        [ex, ey, ez],
        [-ex, ey, ez],
        [-ex, -ey, ez],
        [ex, -ey, ez],
        [ex, -ey, -ez],
        [ex, ey, -ez],
        [-ex, ey, -ez],
        [-ex, -ey, -ez]
    ])
    
    # Convert to homogeneous coordinates for transformation
    vertices_local_homogeneous = np.hstack((vertices_local, np.ones((8, 1))))
    
    # Apply transformation
    vertices_global = (transform @ vertices_local_homogeneous.T).T[:, :3]
    
    # Define edges as pairs of global vertices
    edges_indices = [(0, 1), (1, 2), (2, 3), (3, 0),
                     (4, 5), (5, 6), (6, 7), (7, 4),
                     (0, 5), (1, 6), (2, 7), (3, 4)]
    edges = [(vertices_global[start], vertices_global[end]) for start, end in edges_indices]

    # local_coords = get_local_coords(edges)
    # world_to_local = get_world_to_local(local_coords)
    # xforms = [np.linalg.inv(x) for x in world_to_local]

    lengths = []
    xforms = []
    for edge in edges:
        vector = edge[1] - edge[0]
        length = np.linalg.norm(vector)
        unit = vector / length
        midpoint = edge[0] + (unit * length * .5)
        rotation = trimesh.geometry.align_vectors([0,0,1], unit)
        translation = trimesh.transformations.translation_matrix(midpoint)
        transform = np.dot(translation, rotation)
        lengths.append(length)
        xforms.append(transform)
        
    return edges, lengths, xforms


def normalize(a):
    """Normalizaes vector a.
    """
    if np.linalg.norm(a) == 0:
        # print("ERROR: DIVIDE BY ZERO")
        # exit(0)
        return 0 * a
    return a / np.linalg.norm(a)


# Function to compute rotation matrix to align two vectors
def get_local_coords(edges):
    num_edges = len(edges)
    local_coords = np.zeros((num_edges, 5, 3), dtype=np.float32)

    SAFE_THRESHOLD = 6.1e-3
    CRITICAL_THRESHOLD = 2.5e-4
    THRESHOLD_SQUARED = CRITICAL_THRESHOLD * CRITICAL_THRESHOLD

    for i in range(num_edges):
        local_orig = edges[i][0]
        tail = edges[i][1]
        bone_vec = normalize(tail - local_orig)

        x = bone_vec[0]
        y = bone_vec[1]
        z = bone_vec[2]

        theta = 1 + bone_vec[1]
        theta_alt = x * x + z * z
        
        M = np.zeros((3, 3), dtype=np.float32)

        if theta > SAFE_THRESHOLD or theta_alt > THRESHOLD_SQUARED:
            M[0][1] = -x
            M[1][0] = x
            M[1][1] = y
            M[1][2] = z
            M[2][1] = -z

            if theta <= SAFE_THRESHOLD:
                theta = 0.5 * theta_alt + 0.125 * theta_alt * theta_alt
            
            M[0][0] = 1 - x * x / theta
            M[2][2] = 1 - z * z / theta
            M[0][2] = - x * z / theta
            M[2][0] = - x * z / theta
        else:
            M[0][0] = -1
            M[1][1] = -1
            M[2][2] = 1

        local_coords[i][0] = local_orig
        local_coords[i][1] = tail
        local_coords[i][2:] = M
    
    return local_coords


def get_world_to_local(local_coords):
    """
    local_coords: n_parts, 5, 3

    out: n_parts, 4, 4

    Computes transformation matrices for global coors to local coors.
    Global frame is centered at the origin
    Input to transformation should be a homogenous column vector
    """
    local_transforms = np.zeros(
        [local_coords.shape[0], 4, 4], dtype=np.float32)
    for i in range(local_coords.shape[0]):
        local_frame = local_coords[i]
        local_transforms[i] = get_one_world_to_local(local_frame)
    local_transforms = local_transforms.astype(np.float32)
    return local_transforms


def get_one_world_to_local(local_frame):
    """
    in: (5, 3)
    out: (4, 4)
    """
    u = local_frame[0]
    xyz = local_frame[2:5]
    rot = np.zeros((4, 4), dtype=np.float32)
    rot[:3, :3] = xyz
    rot[3, 3] = 1.0
    tran = np.eye(4, dtype=np.float32)
    tran[:3, 3] = -u
    return np.matmul(rot, tran)


def stitch_imges(out_path, image_paths=None, images=None, adj=100):
    """Stitches images in a row given a list of image paths
    """
    assert (image_paths is None) != (images is None),\
        "Must supply either image_paths or images (not both)"
    if image_paths:
        images = [Image.open(x) for x in image_paths]
    widths, heights = zip(*(i.size for i in images))
    # adj = 100
    total_width = sum(widths) - len(images) * adj
    max_height = max(heights)
    new_im = Image.new('RGBA', (total_width, max_height))
    
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0] - adj
    
    new_im.save(out_path)
