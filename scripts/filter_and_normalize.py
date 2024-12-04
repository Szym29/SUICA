import scanpy as sc
import numpy as np
import anndata as ad
import sys
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from utils import plot_slice

from lightning import pytorch as pl
pl.seed_everything(8848, workers=True)

input_h5ad = "/data/datasets/MOSTA_SUICA/E13.5_E1S1.MOSTA.h5ad"
output_h5ad = "./data/preprocessed_data/PCA-E13.5_E1S1.MOSTA.h5ad"

pca_embd = True

min_genes = 200
min_cells = 3
adata = sc.read_h5ad(input_h5ad)
print("raw shape:", adata.X.shape)
sc.pp.filter_cells(adata, min_counts=200) 
sc.pp.filter_genes(adata, min_counts=3)
print("after filtering:", adata.X.shape)


count = adata.layers['count'].toarray()
column_mask = np.all(count<=200, axis=0)
filtered_count = count[:, column_mask]
print(filtered_count.shape)
new_adata = ad.AnnData(X=filtered_count)
new_adata.obsm["spatial"] = adata.obsm["spatial"]
# adata.X = filtered_count
sc.pp.normalize_total(new_adata, target_sum=200)
print(new_adata.X[new_adata.X > 0].min(), new_adata.X.max())

fig = plot_slice(new_adata.obsm["spatial"], spot_size=1)

fig.savefig("debug.png")

if pca_embd:
    pca = PCA(n_components=16, random_state=0)
    n_cells = new_adata.X.shape[0]
    train_idx, val_idx = train_test_split(list(range(n_cells)), test_size=0.2)
    print(f"{val_idx[:10]=}")
    pca.fit(new_adata.X[train_idx,:]) # make sure PCA fit with only training set
    embd = pca.transform(new_adata.X)
    new_adata.obsm["embeddings"] = embd
 
new_adata.write_h5ad(output_h5ad)