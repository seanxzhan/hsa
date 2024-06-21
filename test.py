import torch


# def compute_bounding_box_and_center(voxel_grid):
#     # Ensure the voxel grid is a binary tensor
#     voxel_grid = (voxel_grid > 0).to(torch.bool)
    
#     # Get the indices of all non-zero elements
#     non_zero_indices = torch.nonzero(voxel_grid)
    
#     # Compute the min and max indices along each dimension
#     min_indices = torch.min(non_zero_indices, dim=0)[0]
#     max_indices = torch.max(non_zero_indices, dim=0)[0]
    
#     # Compute the center of the bounding box
#     center_indices = (min_indices + max_indices).float() / 2.0
    
#     return min_indices, max_indices, center_indices


# def differentiable_bounding_box_and_center(voxel_grid):
#     # Ensure the voxel grid is a float tensor
#     voxel_grid = voxel_grid.to(torch.float32)
    
#     # Create coordinate grids
#     dims = voxel_grid.shape
#     coords = torch.stack(torch.meshgrid([torch.arange(d) for d in dims], indexing='ij'), dim=-1).to(voxel_grid.device)
    
#     # Flatten the voxel grid and coordinates
#     voxel_grid_flat = voxel_grid.view(-1)
#     coords_flat = coords.view(-1, len(dims))
    
#     # Normalize voxel grid to get weights
#     weights = voxel_grid_flat / torch.sum(voxel_grid_flat)
    
#     # Compute weighted sum of coordinates for center
#     center_coords = torch.sum(coords_flat * weights[:, None], dim=0)
    
#     # Compute soft min and max coordinates using weighted sums
#     weighted_coords = coords_flat * weights[:, None]
#     min_coords = torch.sum(weighted_coords, dim=0)
#     max_coords = torch.sum(weighted_coords + (1 - voxel_grid_flat[:, None]) * coords_flat, dim=0)
    
#     return min_coords, max_coords, center_coords


# # Example usage
# voxel_grid = torch.zeros((10, 10, 10), dtype=torch.float32)
# voxel_grid[2:5, 3:6, 1:4] = 1.0  # Example geometry

# min_indices, max_indices, center_indices = compute_bounding_box_and_center(voxel_grid)
# print("Min indices:", min_indices)
# print("Max indices:", max_indices)
# print("Center indices:", center_indices)

# min_indices, max_indices, center_indices = differentiable_bounding_box_and_center(voxel_grid)
# print("Min indices:", min_indices)
# print("Max indices:", max_indices)
# print("Center indices:", center_indices)

def differentiable_bounding_box_and_center(query_points, occupancy_values):
    # Ensure inputs are float tensors
    query_points = query_points.to(torch.float32)
    occupancy_values = occupancy_values.to(torch.float32)
    
    # Normalize occupancy values to get weights
    weights = occupancy_values / (occupancy_values.sum(dim=1, keepdim=True) + 1e-8)

    # print("weights:", weights[0, :, 0])
    # print("points:", query_points[0])

    # Compute the weighted sum of coordinates for the center
    # center_coords = torch.einsum('bnp,bcn->bnp', query_points, weights.transpose(1, 2))
    center_coords = torch.einsum('bcn,bnp->bcp', weights.transpose(1, 2), query_points)

    # print("center coords:", center_coords[0])
    
    # Compute the soft min and max coordinates using log-sum-exp trick
    beta = 10.0  # This is a smoothing parameter
    soft_min_coords = -torch.logsumexp(-query_points[:, :, None, :] * weights[:, :, :, None] * beta, dim=1) / beta
    soft_max_coords = torch.logsumexp(query_points[:, :, None, :] * weights[:, :, :, None] * beta, dim=1) / beta
    
    return soft_min_coords, soft_max_coords, center_coords

# Example usage
batch_size = 2
num_points = 4
query_points = torch.tensor([
    [
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0]
    ],
    [
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0]
    ]
], dtype=torch.float32)

occupancy_values = torch.tensor([
    [
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0]
    ],
    [
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1]
    ]
], dtype=torch.float32)


# print(query_points.shape)
# print(occupancy_values.shape)

min_coords, max_coords, center_coords = differentiable_bounding_box_and_center(query_points, occupancy_values)
# print("Min coordinates:", min_coords)
# print("Max coordinates:", max_coords)
print("query points:", query_points[0])
print("occupancy:", occupancy_values[0, :, 0])
print("center coords:", center_coords[0, 0, :])

# print(occupancy_values[:, :, 0])

