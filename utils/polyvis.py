import numpy as np
import polyscope as ps
import trimesh
# import gpytoolbox as gpy

# NOTE: polyscope doesn't work with headless display

def vis_sdf(sdf, dims, img_path, plot_hist=False, hist_path=None):
    """sdf: numpy array
    """
    ps.init()
    ps.look_at((-2., 2., 2.), (0., 0., 0.))
    ps.set_ground_plane_mode("shadow_only")
    bound_low = (-1., -1., -1.)
    bound_high = (1., 1., 1.)
    ps_grid = ps.register_volume_grid("sdf grid", dims, bound_low, bound_high)
    scalar_vals = np.reshape(sdf, dims)
    # print(np.min(scalar_vals), np.max(scalar_vals))

    if plot_hist:
        import matplotlib.pyplot as plt
        from utils import misc
        plt.hist(sdf.flatten(), bins=50, color='blue')
        plt.title("Histogram of SDF Values")
        plt.xlabel("SDF Value")
        plt.ylabel("Frequency")
        plt.xlim(-1.5, 1.5)
        misc.save_fig(plt, '', hist_path)
        plt.close()
        # plt.show()

    ps_plane = ps.add_scene_slice_plane()
    ps_plane.set_draw_plane(False)
    ps_plane.set_draw_widget(False)

    # add a scalar function on the grid
    ps_grid.add_scalar_quantity("node scalar1", scalar_vals, 
                                defined_on='nodes', 
                                enable_isosurface_viz=True,
                                isosurface_color=[x/255 for x in [43, 151, 242]],
                                isolines_enabled=True,
                                vminmax=(np.min(scalar_vals), np.max(scalar_vals)),
                                enabled=True,
                                cmap='rainbow')
    
    if img_path is not None:
        ps.screenshot(filename=img_path, transparent_bg=False)
    else:
        ps.show()
    ps.remove_all_structures()
    ps.remove_last_scene_slice_plane()


def vis_occ(occ, dims, img_path=None, plot_hist=False, hist_path=None):
    """occ: numpy array
    """
    ps.init()
    ps.look_at((-2., 2., 2.), (0., 0., 0.))
    ps.set_ground_plane_mode("shadow_only")
    bound_low = (-1., -1., -1.)
    bound_high = (1., 1., 1.)
    ps_grid = ps.register_volume_grid("sdf grid", dims, bound_low, bound_high,
                                    #   edge_color=(0, 0, 0),
                                    #   edge_width=1.5,
                                      cube_size_factor=0.1)
    scalar_vals = np.reshape(occ, dims)
    # print(np.min(scalar_vals), np.max(scalar_vals))

    if plot_hist:
        import matplotlib.pyplot as plt
        from utils import misc
        plt.hist(occ.flatten(), bins=50, color='blue')
        plt.title("Histogram of Occupancy Values")
        plt.xlabel("Occupancy Value")
        plt.ylabel("Frequency")
        plt.xlim(-0.5, 1.5)
        misc.save_fig(plt, '', hist_path)
        plt.close()
        # plt.show()

    ps_plane = ps.add_scene_slice_plane()
    ps_plane.set_draw_plane(False)
    ps_plane.set_draw_widget(True)

    # add a scalar function on the grid
    ps_grid.add_scalar_quantity("node scalar1", scalar_vals, 
                                defined_on='nodes', 
                                enable_isosurface_viz=False,
                                # isosurface_color=[x/255 for x in [43, 151, 242]],
                                # isolines_enabled=True,
                                vminmax=(np.min(scalar_vals), np.max(scalar_vals)),
                                enabled=True,
                                cmap='rainbow')
    
    if img_path is not None:
        ps.screenshot(filename=img_path, transparent_bg=False)
    else:
        ps.show()
    # ps.remove_volume_grid("sdf grid")
    ps.remove_all_structures()
    ps.remove_last_scene_slice_plane()


def vis_mesh(vertices, faces, img_path):
    ps.init()
    ps.look_at((-1., 1., 1.), (0., 0., 0.))
    ps.set_ground_plane_mode("shadow_only")

    ps.register_surface_mesh("mesh", vertices, faces)
    ps.screenshot(filename=img_path, transparent_bg=False)
    ps.remove_all_structures()


if __name__ == "__main__":
    expt_name = 'occflexi_2'

    vis_occ(
        np.load(f'results/occflexi/{expt_name}/1000occ.npy'),
        (64, 64, 64),
        f'results/occflexi/{expt_name}/img-1000occ.png',
        plot_hist=True,
        hist_path=f'results/occflexi/{expt_name}/hist-1000occ.png')
    mesh = trimesh.load_mesh(f'results/occflexi/{expt_name}/1000occmesh.obj')
    V, F = mesh.vertices, mesh.faces
    vis_mesh(
        V, F,
        f'results/occflexi/{expt_name}/img-1000occmesh.png')
    vis_sdf(
        np.load(f'results/occflexi/{expt_name}/1000sdf.npy'),
        (32, 32, 32),
        f'results/occflexi/{expt_name}/img-1000sdf.png',
        plot_hist=True,
        hist_path=f'results/occflexi/{expt_name}/hist-1000sdf.png')
    mesh = trimesh.load_mesh(f'results/occflexi/{expt_name}/1000fleximesh.obj')
    V, F = mesh.vertices, mesh.faces
    vis_mesh(
        V, F,
        f'results/occflexi/{expt_name}/img-1000fleximesh.png')

    vis_occ(
        np.load(f'results/occflexi/{expt_name}/outocc.npy'),
        (64, 64, 64),
        f'results/occflexi/{expt_name}/img-outocc.png',
        plot_hist=True,
        hist_path=f'results/occflexi/{expt_name}/hist-outocc.png')
    mesh = trimesh.load_mesh(f'results/occflexi/{expt_name}/outoccmesh.obj')
    V, F = mesh.vertices, mesh.faces
    vis_mesh(
        V, F,
        f'results/occflexi/{expt_name}/img-outoccmesh.png')
    vis_sdf(
        np.load(f'results/occflexi/{expt_name}/outsdf.npy'),
        (32, 32, 32),
        f'results/occflexi/{expt_name}/img-outsdf.png',
        plot_hist=True,
        hist_path=f'results/occflexi/{expt_name}/hist-outsdf.png')
    mesh = trimesh.load_mesh(f'results/occflexi/{expt_name}/outfleximesh.obj')
    V, F = mesh.vertices, mesh.faces
    vis_mesh(
        V, F,
        f'results/occflexi/{expt_name}/img-outfleximesh.png')