def do_boxes_overlap(box1, box2):
    """Check if two boxes overlap or touch. Each box is defined by (min_corner, max_corner)."""
    for i in range(len(box1[0])):
        if box1[1][i] < box2[0][i] or box1[0][i] > box2[1][i]:
            return False
    return True

def find_connected_components(boxes):
    """Find connected components in a set of boxes."""
    n = len(boxes)
    parents = list(range(n))

    def find(x):
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]

    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            parents[rootX] = rootY

    for i in range(n):
        for j in range(i+1, n):
            if do_boxes_overlap(boxes[i], boxes[j]):
                union(i, j)

    return len(set(find(i) for i in range(n)))

# Example usage
boxes = [
    [(0, 0), (1, 1)],  # Box 0
    [(0.9, 0.9), (2, 2)],  # Box 1, overlaps with Box 0
    [(2.1, 2.1), (3, 3)],  # Box 2, separate
    [(3, 0), (4, 1)]  # Box 3, separate
]

print(f"Number of connected components: {find_connected_components(boxes)}")