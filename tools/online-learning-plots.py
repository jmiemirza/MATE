import numpy as np
import matplotlib.pyplot as plt
import os

corruptions = [
    # 'clean',
    'uniform', 'gaussian', 'background', 'impulse', 'upsampling',
    'distortion_rbf', 'distortion_rbf_inv', 'density',
    'density_inc', 'shear', 'rotation', 'cutout', 'distortion', 'occlusion', 'lidar'
]

base_pth = './data/intermediate_accuracy/modelnet/sliding_window'

for corruption in corruptions:

    res = np.load(os.path.join(base_pth, corruption + '.npy'))

    plt.plot(res, label=corruption)
plt.legend()
plt.show()