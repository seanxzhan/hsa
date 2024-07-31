import numpy as np
import matplotlib.pyplot as plt
from utils import ops

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