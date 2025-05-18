import scanpy as sc
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import torch
from rich import print
from omegaconf import OmegaConf
from rich.pretty import pprint

def pprint_config(conf):
    print("[red]Current Configs:[/red]")
    conf_dict = OmegaConf.to_container(conf, resolve=True)
    pprint(conf_dict, expand_all=True)
def construct_subgraph(data, adj,neighbors,idx):
    neighbor_dict = {value: index for index, value in enumerate(neighbors)}
    new_idx = [neighbor_dict[i] for i in idx]
    sub_set_data = data[neighbors]
    sub_set_adj = adj[neighbors][:,neighbors]
    # find the new index of idx in the subgraph
    sub_set_adj = torch.tensor(sub_set_adj.toarray())
    return sub_set_data, sub_set_adj,new_idx

def plot_ST(coordinates, representations=None, spot_size=2, cmap="viridis", title=None):
    if representations is not None and len(representations.shape) == 1:
        # shape = (n_cells,) to (n_cells, 1)
        representations = np.expand_dims(representations, axis=1) 
    assert representations is None or representations.shape[-1] == 3 or representations.shape[-1] == 1
    if coordinates.shape[-1] == 2:
        fig = _plot_slice(coordinates, representations, spot_size, cmap, title)
    elif coordinates.shape[-1] == 3:
        fig = _plot_volume(coordinates, representations, spot_size, cmap, title)
    else:
        raise NotImplementedError
    return fig


def _plot_slice(coordinates, representations=None, spot_size=2, cmap="viridis", title=None):
    fig, ax = plt.subplots()
    ax.axis("equal")
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=14)
    x, y = coordinates[:,0], coordinates[:,1]
    if representations is None:
        ax.scatter(x, y, s=spot_size, cmap=cmap)
    else:
        z = representations
        z_norm = (z - z.min(axis=0)) / (z.max(axis=0) - z.min(axis=0) + 0.00001)
        ax.scatter(x, y, c=z_norm, s=spot_size, cmap=cmap)
    return fig

def _plot_volume(coordinates, representatons=None, spot_size=2, cmap="viridis", title=None):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    if title:
        ax.set_title(title, fontsize=14)
    if representatons is None:
        ax.scatter(coordinates[:,0], coordinates[:,1], coordinates[:,2], s=spot_size, cmap=cmap)
    else:
        z = representatons
        z_norm = (z - z.min(axis=0)) / (z.max(axis=0) - z.min(axis=0) + 0.00001)
        scatter = ax.scatter(coordinates[:,0], coordinates[:,1], coordinates[:,2], c=z_norm, s=spot_size, cmap=cmap)
        fig.colorbar(scatter, ax=ax)
    return fig


def read_anndata(h5ad_file):
    return sc.read_h5ad(h5ad_file)


def read_preprocess_anndata(h5ad_file, min_genes=200, min_cells=3):
    adata = sc.read_h5ad(h5ad_file)
    raw_size = adata.X.shape
    sc.pp.filter_cells(adata, min_counts=min_genes) # filter cells containing less than #min_genes genes
    sc.pp.filter_genes(adata, min_counts=min_cells) # filter genes appearing in less than #min_cells cells
    new_size = adata.X.shape
    print(f"Filtering ST data from {raw_size} to {new_size} ...")
    return adata


def _cosine_similarity(y_true, y_pred, mask=False):
    if mask:
        y_true, y_pred = y_true[y_true>0], y_pred[y_true>0]
    numerator = np.sum(y_true * y_pred, axis=-1)  
    denominator = np.sqrt(np.sum(y_true ** 2, axis=-1)) * np.sqrt(np.sum(y_pred ** 2, axis=-1))  
    pixelwise_cosine = numerator / (denominator + 0.00001)
    return pixelwise_cosine.mean() 

def _spectral_angle_mapper(y_true, y_pred, mask=False):
    if mask:
        y_true, y_pred = y_true[y_true>0], y_pred[y_true>0]
    numerator = np.sum(y_true * y_pred, axis=-1)  
    denominator = np.sqrt(np.sum(y_true ** 2, axis=-1)) * np.sqrt(np.sum(y_pred ** 2, axis=-1))  
    pixelwise_cosine = numerator / denominator
    cos_theta = np.clip(pixelwise_cosine, -1.0, 1.0) 
    sam_angle = np.rad2deg(np.arccos(cos_theta))
    return sam_angle.mean()

def _spearman_r(y_true, y_pred, mask=False):
    if mask:
        corrs = np.array([np.nan_to_num(spearmanr(y_true[i][y_true[i]>0], y_pred[i][y_true[i]>0]).statistic) for i in range(y_pred.shape[0])])
    else:
        corrs = np.array([np.nan_to_num(spearmanr(y_true[i], y_pred[i]).statistic) for i in range(y_pred.shape[0])])
    return corrs.mean()

def _pearson_r(y_true, y_pred, mask=False):
    if mask:
        corrs = np.array([np.nan_to_num(pearsonr(y_true[i][y_true[i]>0], y_pred[i][y_true[i]>0]).statistic) for i in range(y_pred.shape[0])])
    else:
        corrs = np.array([np.nan_to_num(pearsonr(y_true[i], y_pred[i]).statistic) for i in range(y_pred.shape[0])])
    return corrs.mean()

# Intersection over Union of zero-map
def _IoU(y_true, y_pred):
    zero_map_A, zero_map_B = (y_true == 0), (y_pred == 0)
    intersection = np.logical_and(zero_map_A, zero_map_B).sum()
    union = np.logical_or(zero_map_A, zero_map_B).sum()

    iou = intersection / union if union != 0 else 0
    return iou

# Intersection over Union of non-zero-map
def _support_recovery_rate(y_true, y_pred):
    one_map_A, one_map_B = (y_true > 0), (y_pred > 0)
    intersection = np.logical_and(one_map_A, one_map_B).sum()
    union = np.logical_or(one_map_A, one_map_B).sum()
    iou = intersection / union if union != 0 else 0
    return iou

def _masked_MSE(y_true, y_pred):
    return mean_squared_error(y_true[y_true>0], y_pred[y_true>0])

def _masked_MAE(y_true, y_pred):
    return mean_absolute_error(y_true[y_true>0], y_pred[y_true>0])

def metrics(y_true, y_pred, prefix="val", fast=False):
    if isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
    scores = {
        f"{prefix}/mean_absolute_error": mean_absolute_error(y_true, y_pred),
        f"{prefix}/mean_absolute_error_mask": _masked_MAE(y_true, y_pred),
        f"{prefix}/mean_squared_error": mean_squared_error(y_true, y_pred),
        f"{prefix}/mean_squared_error_mask": _masked_MSE(y_true, y_pred),
        f"{prefix}/root_mean_squared_error": root_mean_squared_error(y_true, y_pred),
        f"{prefix}/cosine_similarity": _cosine_similarity(y_true, y_pred),
        f"{prefix}/cosine_similarity_mask": _cosine_similarity(y_true, y_pred, mask=True),
        f"{prefix}/sam": _spectral_angle_mapper(y_true, y_pred),
        # f"{prefix}/sam_mask": _spectral_angle_mapper(y_true, y_pred, mask=True),
        f"{prefix}/iou": _IoU(y_true, y_pred),
    }
    if not fast:
        slow_metrics = {
            # may take some time
            #f"{prefix}/r2_score": r2_score(y_true, y_pred),
            f"{prefix}/pearsonr": _pearson_r(y_true, y_pred),
            f"{prefix}/spearmanr": _spearman_r(y_true, y_pred), 
            f"{prefix}/pearsonr_mask": _pearson_r(y_true, y_pred, mask=True),
            f"{prefix}/spearmanr_mask": _spearman_r(y_true, y_pred, mask=True), 
        }
        scores.update(slow_metrics)

    return scores


if __name__ == "__main__":
    np.random.seed(0)

    # Example arrays
    A = np.random.rand(10000, 20000)
    B = np.random.rand(10000, 20000)

    print(_spectral_angle_mapper(A,B))
    print(_cosine_similarity(A,B))