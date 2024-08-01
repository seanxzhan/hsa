import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import ops, polyvis


def gaussian_kernel_3d(kernel_size=3, sigma=1.0):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy, zz = np.meshgrid(ax, ax, ax)
    
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy) + np.square(zz)) / np.square(sigma))
    kernel = kernel / np.sum(kernel)
    
    return torch.tensor(kernel, dtype=torch.float32)

def apply_3d_gaussian_blur(input_tensor, kernel_size=3, sigma=1.0):
    kernel = gaussian_kernel_3d(kernel_size, sigma)
    kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
    
    input_tensor = input_tensor.unsqueeze(0)  # Adding batch and channel dimensions
    blurred_tensor = torch.nn.functional.conv3d(
        input_tensor, kernel, padding=kernel_size//2)
    
    return blurred_tensor.squeeze(1)  # Removing batch and channel dimensions


def test_conv():
    occ = np.load('results/flexi-occsdf/bin2sdf/43941_gt_occ.npy')
    occ = occ.reshape([32]*3)
    # polyvis.vis_occ(occ, [32]*3, 'occ_before.png')
    occ = torch.from_numpy(occ)
    pseudo_sdf = apply_3d_gaussian_blur(occ)
    np.save('psuedo_sdf.npy', pseudo_sdf.numpy())
    polyvis.vis_occ(pseudo_sdf, [32]*3, 'occ_after.png')

test_conv()


def test_distr():
    # Generate a sample signal
    # x = np.random.beta(a=2, b=2, size=1000)  # Example data between 0 and 1
    x = np.load('results/occflexi/occflexi_1/bin2sdf_torch_6/1000occ.npy').flatten()
    y = np.load('results/occflexi/occflexi_1/bin2sdf_torch_6/outocc.npy').flatten()

    # Apply transformations
    epsilon = 1e-6
    log_transformed = np.log(x + epsilon)
    exp_transformed = np.exp(x) - 1
    sigmoid_transformed = 1 / (1 + np.exp(-10 * (x - 0.5)))
    power_transformed = np.power(x, 2)
    arsinh_transformed = np.arcsinh(1.15*x)
    smoothed = ops.smoothing_sinh_np(x)


    # Plot the transformations
    plt.figure(figsize=(24, 16))

    nrows = 4

    plt.subplot(nrows, 2, 1)
    plt.hist(x, bins=50, color='blue', alpha=0.7)
    plt.xlim((-0.25, 1.5))
    plt.ylim((0, 250000))
    plt.title('Original Data')

    plt.subplot(nrows, 2, 2)
    plt.hist(log_transformed, bins=50, color='green', alpha=0.7)
    plt.xlim((-0.25, 1.5))
    plt.ylim((0, 250000))
    plt.title('Log Transformation')

    plt.subplot(nrows, 2, 3)
    plt.hist(exp_transformed, bins=50, color='red', alpha=0.7)
    plt.xlim((-0.25, 1.5))
    plt.ylim((0, 250000))
    plt.title('Exponential Transformation')

    plt.subplot(nrows, 2, 4)
    plt.hist(sigmoid_transformed, bins=50, color='purple', alpha=0.7)
    plt.xlim((-0.25, 1.5))
    plt.ylim((0, 250000))
    plt.title('Sigmoid Transformation')

    plt.subplot(nrows, 2, 5)
    plt.hist(power_transformed, bins=50, color='orange', alpha=0.7)
    plt.xlim((-0.25, 1.5))
    plt.ylim((0, 250000))
    plt.title('Power Transformation')

    plt.subplot(nrows, 2, 6)
    plt.hist(arsinh_transformed, bins=50, color='brown', alpha=0.7)
    plt.xlim((-0.25, 1.5))
    plt.ylim((0, 250000))
    plt.title('Arsinh Transformation')

    plt.subplot(nrows, 2, 7)
    plt.hist(smoothed, bins=50, color='magenta', alpha=0.7)
    plt.xlim((-0.25, 1.5))
    plt.ylim((0, 250000))
    plt.title('Smoothed Transformation')

    plt.subplot(nrows, 2, 8)
    plt.hist(y, bins=50, color='gray', alpha=0.7)
    plt.xlim((-0.25, 1.5))
    plt.ylim((0, 250000))
    plt.title('Target Transformation')

    plt.tight_layout()
    plt.show()