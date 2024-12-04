import scanpy as sc
from scipy.interpolate import griddata
from kornia.utils import create_meshgrid
import anndata as ad
from einops import rearrange
import numpy as np
import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from utils import plot_slice

input_h5ad = "data/preprocessed_data/E10.5_E1S1.MOSTA.h5ad"
output_h5ad = "data/preprocessed_data/resampled-E10.5_E1S1.MOSTA.h5ad"

adata = sc.read_h5ad(input_h5ad)


def get_grid_locations(coordinates, side_length=200):
    grid = create_meshgrid(side_length, side_length, normalized_coordinates=True, dtype=float) # 1, H, W, 2
    grid = rearrange(grid[0], "H W C -> (H W) C").numpy() # N, 2  value=-1 to 1

    x_min, x_max = coordinates[:,0].min(), coordinates[:,0].max()
    y_min, y_max = coordinates[:,1].min(), coordinates[:,1].max()
    grid[:,0] = x_min + (grid[:,0] + 1) / 2 * (x_max - x_min)
    grid[:,1] = y_min + (grid[:,1] + 1) / 2 * (y_max - y_min)
    return grid
    

representations = adata.X
coordinates = adata.obsm["spatial"]
print(representations.shape)
resampled_coords = get_grid_locations(coordinates)

pred_val = griddata(coordinates, representations, resampled_coords, method="linear")
print(pred_val.shape)
row_mask = np.any(np.isnan(pred_val), axis=1) # check nan
valid_raw = pred_val[~row_mask, :]
valid_coords = resampled_coords[~row_mask, :]
print(valid_raw.shape)
fig = plot_slice(valid_coords, spot_size=0.01)
fig.savefig("resampled.png")

new_adata = ad.AnnData(X=valid_raw)
new_adata.obsm["spatial"] = valid_coords

new_adata.write_h5ad(output_h5ad)