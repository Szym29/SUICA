from scipy.interpolate import griddata
from rich import print
from lightning import pytorch as pl
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import anndata as ad
import importlib


from datasets import ST2D
from utils import metrics


def fit_griddata(configs):
    dataset_configs = configs.dataset
    pipeline_configs = configs.pipeline
    pl.seed_everything(pipeline_configs.optimization.seed, workers=True) # fix seed globally

    # dataset configuration
    dataset_class = getattr(importlib.import_module("datasets"), dataset_configs.type)
    dataset = dataset_class(**dataset_configs)
    
    # split from data_file
    if dataset_configs.type == "ST2D":
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=dataset_configs.val_proportion)
        print(f"{val_idx[:10]=}") # check whether seed works
    else:
        all_idx = list(range(len(dataset)))
        val_idx = all_idx[::4]
        train_idx = list(set(all_idx) - set(val_idx))
    train_dataset, val_dataset = Subset(dataset, train_idx), Subset(dataset, val_idx)
    
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=len(train_dataset), num_workers=8, drop_last=False)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=len(val_dataset), num_workers=8, drop_last=False)
        
    for sample_train in train_dataloader:
        coords_train, raw_train = sample_train["coordinates"].numpy(),  sample_train["raw_representations"].numpy()
    for sample_val in val_dataloader:
        coords_val, raw_val = sample_val["coordinates"].numpy(), sample_val["raw_representations"].numpy()

    pred_val = griddata(coords_train, raw_train, coords_val, method=pipeline_configs.interp_method, fill_value=0)

    scores = metrics(raw_val, pred_val, "interp")
    print(scores)

    adata = ad.AnnData(X=raw_val)
    adata.obsm["spatial"] = coords_val
    adata.obsm["reconstructed_raw"] = pred_val
    adata.write_h5ad("grid-val.h5ad")
