import scanpy as sc
import sys
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from utils import metrics
from lightning import pytorch as pl
pl.seed_everything(8848, workers=True)

adata_raw = sc.read_h5ad("data/preprocessed_data/PCA-E13.5_E1S1.MOSTA.h5ad")
adata_fit = sc.read_h5ad("logs/inr_pca_13.5/lightning_logs/version_0/reconstructed-E13.5_E1S1.MOSTA.h5ad")

n_cells = adata_raw.X.shape[0]
train_idx, val_idx = train_test_split(list(range(n_cells)), test_size=0.2)
pca = PCA(n_components=16, random_state=0)
pca.fit(adata_raw.X[train_idx,:])

raw_val = adata_fit.X
pred_val = pca.inverse_transform(adata_fit.obsm["fitted_embd"])

scores = metrics(raw_val, pred_val)
print(scores)