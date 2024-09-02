import torch
import torch.multiprocessing as mp

def process_shape(shape):
    # Example processing: scaling the shape
    scale_factor = 2.0
    processed_shape = shape * scale_factor
    return processed_shape

def process_batch(batch_of_shapes):
    # Number of processes (you can set this to the number of CPU cores available)
    num_processes = mp.cpu_count()
    print(num_processes)

    # Create a multiprocessing Pool
    with mp.Pool(processes=num_processes) as pool:
        # Map the processing function to the batch of shapes
        results = pool.map(process_shape, batch_of_shapes)

    return results

if __name__ == "__main__":
    # Example batch of shapes (let's assume each shape is a tensor of size (3, 3, 3))
    batch_of_shapes = [torch.randn(3, 3, 3) for _ in range(10)]

    # Process the batch in parallel
    processed_shapes = process_batch(batch_of_shapes)

    # Print the processed shapes
    for i, shape in enumerate(processed_shapes):
        print(f"Processed shape {i}:\n{shape}\n")