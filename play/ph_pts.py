import itertools
import numpy as np

def distance(p1, p2):
    """Compute Euclidean distance between two points."""
    return np.sqrt(sum((p1 - p2) ** 2))

def create_vietoris_rips_complex(points, max_edge_length):
    """Generate edges for the Vietoris-Rips complex given a set of points."""
    edges = []
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points[i+1:], i+1):
            if distance(p1, p2) <= max_edge_length:
                edges.append([i, j])
    return edges

def find_connected_components(edges, num_points):
    """Find connected components in the graph formed by edges."""
    parents = list(range(num_points))  # Each point is initially its own parent

    def find(x):
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]

    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            parents[rootX] = rootY

    for edge in edges:
        x, y = edge
        union(x, y)

    # Finding unique parents will give us unique connected components
    return len(set(find(x) for x in range(num_points)))

# Example usage
points = np.array([
    [0, 0],
    [1, 0],
    [0.5, 0.5],
    [0, 2],
    [1.5, 1.5],
    [2.0, 1.0]
])

max_edge_length = 1

# Construct Vietoris-Rips complex
edges = create_vietoris_rips_complex(points, max_edge_length)
print(edges)

# Find connected components (0-dimensional homology)
num_components = find_connected_components(edges, len(points))

print(f"Number of connected components: {num_components}")

import matplotlib.pyplot as plt
from utils import misc

# Creating the scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], color='blue', marker='o')
for e in edges:
    plt.plot([points[e[i]][0] for i in [0,1]],
             [points[e[i]][1] for i in [0,1]],
             color='orange')
plt.xlim(-0.2, 2.2)
plt.ylim(-0.2, 2.2)
plt.grid(True)  # Show grid

misc.save_fig(plt, f'connected components: {num_components}',
              'play/tmp/ph_pts.png')