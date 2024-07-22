import numpy as np
import polyscope as ps
import gpytoolbox as gpy

# NOTE: polyscope doesn't work with headless display

def vis_sdf(sdf, dims, img_path):
    """sdf: torch tensor
    """
    ps.init()
    ps.look_at((-2., 2., 2.), (0., 0., 0.))
    ps.set_ground_plane_mode("shadow_only")
    bound_low = (-1., -1., -1.)
    bound_high = (1., 1., 1.)
    ps_grid = ps.register_volume_grid("sdf grid", dims, bound_low, bound_high)
    scalar_vals = np.reshape(sdf, dims)
    print(np.min(scalar_vals), np.max(scalar_vals))

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
    
    ps.screenshot(filename=img_path, transparent_bg=False)


def vis_mesh(vertices, faces, img_path):
    ps.init()
    ps.look_at((-2., 2., 2.), (0., 0., 0.))
    ps.set_ground_plane_mode("shadow_only")

    mesh = ps.register_surface_mesh("mesh", vertices, faces)
    ps.screenshot(filename=img_path, transparent_bg=False)