import scipy 
import numpy as np
import os

mat_path = "/data/datasets/HRSS/raw"


mat = scipy.io.loadmat(os.path.join(mat_path, "Salinas_corrected.mat"))
print(mat.keys())
image = mat["salinas_corrected"]
print(image.shape)

image = np.flipud(image)
image = np.transpose(image, (1, 0, 2))

np.savez('/data/datasets/HRSS/Salinas.npz', raw=image)
