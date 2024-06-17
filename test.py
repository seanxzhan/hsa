import torch


def compute_bounding_box_and_center(voxel_grid):
    # Ensure the voxel grid is a binary tensor
    voxel_grid = (voxel_grid > 0).to(torch.bool)
    
    # Get the indices of all non-zero elements
    non_zero_indices = torch.nonzero(voxel_grid)
    
    # Compute the min and max indices along each dimension
    min_indices = torch.min(non_zero_indices, dim=0)[0]
    max_indices = torch.max(non_zero_indices, dim=0)[0]
    
    # Compute the center of the bounding box
    center_indices = (min_indices + max_indices).float() / 2.0
    
    return min_indices, max_indices, center_indices


# def differentiable_bounding_box_and_center(voxel_grid):
#     # Ensure the voxel grid is a binary tensor
#     voxel_grid = voxel_grid.to(torch.float32)
    
#     # Create coordinate grids
#     dims = voxel_grid.shape
#     coords = torch.stack(torch.meshgrid([torch.arange(d) for d in dims]), dim=-1).to(voxel_grid.device)
    
#     # Compute weighted sums for each coordinate axis
#     total_weight = voxel_grid.sum()
    
#     min_coords = []
#     max_coords = []
#     center_coords = []
    
#     for i in range(len(dims)):
#         coord_axis = coords[..., i]
        
#         # Weighted sums to approximate min and max (approximations)
#         weighted_sum = (voxel_grid * coord_axis).sum() / total_weight
#         weighted_square_sum = (voxel_grid * coord_axis**2).sum() / total_weight
        
#         # Approximate min and max
#         coord_min = 2 * weighted_sum - (2 * weighted_square_sum - weighted_sum**2).sqrt()
#         coord_max = 2 * weighted_sum + (2 * weighted_square_sum - weighted_sum**2).sqrt()
        
#         min_coords.append(coord_min)
#         max_coords.append(coord_max)
        
#         # Center is simply the weighted sum
#         center_coords.append(weighted_sum)
    
#     min_coords = torch.stack(min_coords)
#     max_coords = torch.stack(max_coords)
#     center_coords = torch.stack(center_coords)
    
#     return min_coords, max_coords, center_coords


def differentiable_bounding_box_and_center(voxel_grid, temperature=1.0):
    # Ensure the voxel grid is a float tensor
    voxel_grid = voxel_grid.to(torch.float32)
    
    # Create coordinate grids
    dims = voxel_grid.shape
    coords = torch.stack(torch.meshgrid([torch.arange(d) for d in dims], indexing='ij'), dim=-1).to(voxel_grid.device)
    
    # Flatten the voxel grid and coordinates
    voxel_grid_flat = voxel_grid.view(-1)
    coords_flat = coords.view(-1, len(dims))
    
    # Compute weights for softmax
    weights = torch.softmax(voxel_grid_flat / temperature, dim=0)
    
    # Compute soft min and max coordinates
    weighted_coords = coords_flat * weights[:, None]
    
    soft_min_coords = torch.sum(weighted_coords, dim=0)
    soft_max_coords = torch.sum(weighted_coords, dim=0) + (coords_flat.max(dim=0)[0] - coords_flat.min(dim=0)[0])
    
    # Compute center coordinates
    center_coords = (soft_min_coords + soft_max_coords) / 2
    
    return soft_min_coords, soft_max_coords, center_coords


# Example usage
voxel_grid = torch.zeros((10, 10, 10), dtype=torch.float32)
voxel_grid[2:5, 3:6, 1:4] = 1.0  # Example geometry

min_indices, max_indices, center_indices = compute_bounding_box_and_center(voxel_grid)
print("Min indices:", min_indices)
print("Max indices:", max_indices)
print("Center indices:", center_indices)

min_indices, max_indices, center_indices = differentiable_bounding_box_and_center(voxel_grid)
print("Min indices:", min_indices)
print("Max indices:", max_indices)
print("Center indices:", center_indices)