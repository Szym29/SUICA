from sklearn.neighbors import kneighbors_graph 
import numpy as np
import scanpy as sc

adata = sc.read("data/aligned_data/E11.5_E1S1.MOSTA.h5ad")

adata_embd = sc.read('data/embedded_data/E11.5_E1S1.MOSTA.h5ad')
adata.obsm["embeddings"] = adata_embd.obsm["embeddings"]

sc.pp.neighbors(adata, use_rep='embeddings', n_neighbors=25, metric='euclidean')
sc.tl.umap(adata)
sc.pl.umap(adata, color='annotation', save='umap_plot.png')