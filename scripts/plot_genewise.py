import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from utils import plot_ST
import matplotlib.pyplot as plt
import pickle
import scanpy as sc

CMAP = "PuBu"

DEST = "9.5_genewise"

gene_names = "/data/datasets/data_SUICA/raw_data/MOSTA_SUICA/gene_names/9.5.pkl"

methods = {
    "STAGE": "/data/datasets/data_SUICA/STAGE/stage_E9.5_E1S1.MOSTA.h5ad",
    "AE+FFN": "logs/AE+FFN/9.5/lightning_logs/version_0/reconstructed-val.h5ad",
    #"SIREN": "logs/SIREN/9.5/lightning_logs/version_0/reconstructed-val.h5ad",
    "FFN": "logs/FFN/9.5/lightning_logs/version_0/reconstructed-val.h5ad",
    #"PCA": "logs/non-learning/9.5/pca-val.h5ad",
    #"grid": "logs/non-learning/9.5/grid-val.h5ad"
}


# methods = {
#     "STAGE": "/data/datasets/data_SUICA/STAGE/stage_E16.5_E1S1.MOSTA.h5ad",
#     "AE+FFN": "logs/AE+FFN/16.5/lightning_logs/version_0/reconstructed-val.h5ad",
#     "SIREN": "logs/SIREN/16.5/lightning_logs/version_0/reconstructed-val.h5ad",
#     "FFN": "logs/FFN/16.5/lightning_logs/version_0/reconstructed-val.h5ad",
#     "PCA": "logs/non-learning/16.5/pca-val.h5ad",
#     "grid": "logs/non-learning/16.5/grid-val.h5ad"
# }

if not os.path.exists(DEST):
    os.makedirs(DEST, exist_ok=True)

with open(gene_names, "rb") as f:
    gene_list = pickle.load(f)

adata_dict = {x: sc.read_h5ad(methods[x]) for x in methods}

_, first_adata = next(iter(adata_dict.items()))
gt = first_adata.X
spatial = first_adata.obsm["spatial"]

recon = {}
for method, adata in adata_dict.items():
    if "fitted_raw" in adata.obsm_keys():
        pred = adata.obsm["fitted_raw"]
    elif "reconstructed_raw" in adata.obsm_keys():
        pred = adata.obsm["reconstructed_raw"]
    elif "recons" in adata.obsm_keys():
        pred = adata.obsm["recons"]
    else:
        raise NotImplementedError
    recon[method] = pred


# left-bottom: 0 ae+ffn
# right-bottom: 1 siren
# left-up: 2 ffn
# right-up: 3 gt

# spatial_0 = spatial.copy()
# spatial_1 = spatial.copy()
# spatial_2 = spatial.copy()
# spatial_3 = spatial.copy()

# spatial_1[:,0] = spatial[:,0] + 3.0
# spatial_2[:,1] = spatial[:,1] + 3.0
# spatial_3 = spatial_3 + 3.0

# spatial = np.concatenate((spatial_0, spatial_1, spatial_2, spatial_3), axis=0)
# print(spatial.shape)

# for i in range(gt.shape[-1]):
#     slices = [np.expand_dims(pred[:, i], axis=1) for pred in recon]
#     slices.append(np.expand_dims(gt[:, i], axis=1))
#     group = np.concatenate(slices, axis=0)
#     print(group.shape)
#     fig = plot_ST(spatial, group[:,0], cmap=CMAP, title=f"{i}")
#     fig.savefig("test.png", dpi=200)
#     plt.close('all')


# iter genes
for i in range(len(gene_list)):
    gene = gene_list[i]
    gt_slice = gt[:, i]
    slices = {method: pred[:, i] for method, pred in recon.items()}
    print(i)

    fig_gt = plot_ST(spatial, gt_slice, title=f"GT-{i}-{gene}", cmap=CMAP)
    fig_list = [fig_gt]
    for method, slice in slices.items():
        fig = plot_ST(spatial, slice, title=f"{method}-{i}-{gene}", cmap=CMAP)
        fig_list.append(fig)

    combined_fig, axes = plt.subplots(1, len(fig_list), figsize=(5 * len(fig_list), 5), squeeze=False)
    for j, fig in enumerate(fig_list):
        fig.canvas.draw()
        ax_img = fig.canvas.buffer_rgba()  
        axes[0, j].imshow(ax_img)
        axes[0, j].axis("off")  
    plt.tight_layout()
    combined_fig.savefig(f"{DEST}/{i}-{gene}.png")
    plt.close('all')