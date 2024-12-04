from sklearn.neighbors import kneighbors_graph
import scanpy as sc
import scipy.sparse as sp
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from utils import plot_ST

adata_ae = sc.read("./logs/AE/16.5/lightning_logs/version_0/embedded-all.h5ad")
adata_gae = sc.read("/data/datasets/data_SUICA/E16.5_all_GAEFFN.h5ad")

from scipy.stats import entropy

def channel_wise_entropy(arr, bins=256):
    """
    Calculate channel-wise entropy for a numpy array of shape (N, d).
    
    Parameters:
    - arr: numpy array of shape (N, d)
    - bins: number of bins to estimate the probability distribution (default: 256)
    
    Returns:
    - entropies: numpy array of shape (d,) containing the entropy for each channel
    """
    N, d = arr.shape
    entropies = np.zeros(d)
    
    # Calculate entropy for each channel (i.e., column)
    for i in range(d):
        # Get histogram counts and bin edges
        counts, _ = np.histogram(arr[:, i], bins=bins, density=True)
        counts += 1e-9  # Add a small value to avoid log(0)
        
        # Normalize counts to get probabilities
        probabilities = counts / np.sum(counts)
        
        # Calculate entropy using scipy's entropy function
        entropies[i] = entropy(probabilities)
    
    return entropies


def node_wise_graph_total_variation(A: csr_matrix, X: np.ndarray) -> np.ndarray:
    """
    Compute the node-wise graph total variation of a multi-dimensional signal X on a graph with adjacency matrix A.

    Parameters:
    - A: csr_matrix, the adjacency matrix of the graph (n x n).
    - X: np.ndarray, the multi-dimensional signal on the graph (n x d).

    Returns:
    - np.ndarray, the node-wise total variation of the multi-dimensional signal on the graph (length n).
    """
    # Ensure that X has shape (n, d)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    
    n = A.shape[0]
    node_tv = np.zeros(n)
    
    # Get the indices of non-zero elements in the adjacency matrix (i.e., the edges)
    row, col = A.nonzero()
    
    # Compute the L2 norm between feature vectors of connected nodes
    differences = np.linalg.norm(X[row] - X[col], axis=1)
    # differences = np.cos(X[row], X[col])

    # Accumulate the total variation for each node
    for i, (r, c) in enumerate(zip(row, col)):
        node_tv[r] += differences[i]
        node_tv[c] += differences[i]
    
    return node_tv

for adata, method in zip([adata_ae, adata_gae], ["AE", "GAE"]):

    points = adata.obsm["spatial"]
    signal = adata.obsm["embeddings"]

    if sp.issparse(signal):
        signal = signal.toarray()
    
    graph = kneighbors_graph(points, 4, mode="connectivity")

    #graph = exp_csr_matrix(graph)
    gtv = node_wise_graph_total_variation(graph, signal)

    print(f"{method:>3} GTV (k=4): {gtv.mean():.2f}")

    mask = gtv < np.percentile(gtv, 95)



    plt.hist(gtv[mask], bins=500)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {method}')
    plt.grid(True)
    #plt.xlim(0,500)
    plt.savefig(f"{method}-histogram.png")
    plt.close()
    fig = plot_ST(points[mask], gtv[mask], cmap="inferno")
    fig.savefig(f"{method}.png")
    plt.close()

    entropies = channel_wise_entropy(signal)
    variances = np.var(signal, axis=0)
    
    print(entropies.mean(), variances.mean())

