import scanpy as sc
import anndata as ad
from lightning import pytorch as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score
pl.seed_everything(8848, workers=True) # fix seed globally


class ari_resolver:
    def __init__(self, adata, adata_org,cell_type_key,pred_key = 'reconstructed_raw'):
        self.adata = sc.read(adata)
        self.adata_org = sc.read(adata_org)
        self.cell_type = cell_type_key
        self.pred_key = pred_key
        self.adata.obs = self.adata_org.obs
        train_idx, val_idx = train_test_split(list(range(len(self.adata_org))), test_size=0.2)
        self.adata_val = self.adata[val_idx]
        self.adata_org_val = self.adata_org[val_idx]
        self.ari = None
        
    def get_ari_all_recons_raw(self,neighbors=30,resolutoin=1):
        temp = self.adata.copy()
        new_temp = ad.AnnData(X=temp.obsm[self.pred_key], obs=temp.obs,var=temp.var)
        sc.pp.neighbors(new_temp,n_neighbors=neighbors)
        sc.tl.umap(new_temp)
        sc.tl.leiden(new_temp,resolution=resolutoin)
        ari = adjusted_rand_score(new_temp.obs['leiden'],new_temp.obs[self.cell_type])
        sc.pl.umap(new_temp,color=['leiden',self.cell_type],title=['reconstructed_raw',f'ARI={ari}'],ncols=1)
        print('ARI for all reconstructed raw data:',ari)

    def get_ari_val_recons_raw(self,neighbors=30,resolutoin=1):
        temp = self.adata_val.copy()
        new_temp = ad.AnnData(X=temp.obsm[self.pred_key], obs=temp.obs,var=temp.var)
        sc.pp.neighbors(new_temp,n_neighbors=neighbors)
        sc.tl.umap(new_temp)
        sc.tl.leiden(new_temp,resolution=resolutoin)
        ari = adjusted_rand_score(new_temp.obs['leiden'],new_temp.obs[self.cell_type])
        sc.pl.umap(new_temp,color=['leiden',self.cell_type],title=['reconstructed_raw',f'ARI={ari}'],ncols=1)
        print('ARI for validation reconstructed raw data:',ari)