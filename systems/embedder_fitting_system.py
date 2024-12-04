import torch
from torch.utils.data import DataLoader, Subset
import lightning as L
from lightning import pytorch as pl
from sklearn.model_selection import train_test_split
from rich import print
import anndata as ad
import os
import numpy as np
from omegaconf import ListConfig
import importlib

from datasets import ST2D, GraphST2D,GraphST3D
from networks import AE, GAE
from utils import metrics, plot_ST, construct_subgraph

class ExtraVariableCallback(pl.Callback):
    def __init__(self, extra_variable):
        self.extra_variable = extra_variable  # Store your extra variable

    def on_train_start(self, trainer, pl_module):
        pl_module.extra_variable = self.extra_variable
        print(f"Extra variable passed to LightningModule for training.")

    def on_validation_start(self, trainer, pl_module):
        pl_module.extra_variable = self.extra_variable
        print(f"Extra variable passed to LightningModule for validation.")

    def on_predict_start(self, trainer, pl_module):
        pl_module.extra_variable = self.extra_variable
        print(f"Extra variable passed to LightningModule for prediction.")

    # These methods are required by some stateful callbacks to avoid the "Expected a parent" error
    def state_dict(self):
        # Return an empty dict since no state needs to be saved
        return {}
    
class EmbedderFittingSystem(L.LightningModule):
    def __init__(self, configs, val_pca=None):
        super().__init__()

        self.save_hyperparameters()

        self.pipeline_configs = configs
        network_configs = self.pipeline_configs.embedder

        if network_configs.model == "AE":
            self.GNN = False
            self.fitting_model = AE(
                dim_in=network_configs.dim_in,
                dim_hidden=network_configs.dim_hidden,
                dim_latent=network_configs.dim_latent
            )
        elif network_configs.model == "GAE":
            self.GNN = True
            self.fitting_model = GAE(
                dim_in=network_configs.dim_in,
                dim_hidden=network_configs.dim_hidden,
                dim_latent=network_configs.dim_latent
            )
        else:
            raise NotImplementedError

        self.val_pca = val_pca
        self.output_cache = []
        
    
    def forward(self, x):
        recon, embd = self.fitting_model(x)
        return recon, embd
    
    def training_step(self, batch, batch_idx):
        if self.GNN:
            idx, semantic_neighbors = batch["idx"],batch["semantic_neighbors"]
            extra_info = self.extra_variable
            adj = extra_info["adj_train"]
            raw_rep = extra_info["raw_rep_train"]
            content_neighbours = list(semantic_neighbors.cpu().numpy().flatten())
            neighbourhoods = list(set(content_neighbours))
            sub_set_y_raw, sub_set_adj,sub_set_idx = construct_subgraph(raw_rep, adj, neighbourhoods,idx.cpu().numpy())
            sub_set_y_raw = sub_set_y_raw.cuda()
            sub_set_adj = sub_set_adj.cuda()
            loss, _, _ = self.fitting_model.forward_loss(sub_set_y_raw,sub_set_adj,sub_set_idx)
            del adj
            del raw_rep
        else:
            y_raw = batch["raw_representations"]
            loss, _, _ = self.fitting_model.forward_loss(y_raw)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        if self.GNN:
            y_raw, x,idx,semantic_neighbors = batch["raw_representations"], batch["coordinates"], batch["idx"],batch["semantic_neighbors"]
            extra_info = self.extra_variable
            adj = extra_info["adj_val"]
            raw_rep = extra_info["raw_rep_val"]
            content_neighbours = list(semantic_neighbors.cpu().numpy().flatten()) 
            neighbourhoods = list(set(content_neighbours))
            sub_set_y_raw, sub_set_adj,sub_set_idx = construct_subgraph(raw_rep, adj, neighbourhoods,idx.cpu().numpy())
            sub_set_y_raw = sub_set_y_raw.cuda()
            sub_set_adj = sub_set_adj.cuda()
            loss, y_hat, _ = self.fitting_model.forward_loss(sub_set_y_raw,sub_set_adj,sub_set_idx)

        else:
            y_raw, x = batch["raw_representations"], batch["coordinates"]
            loss, y_hat, _ = self.fitting_model.forward_loss(y_raw)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.output_cache.append({
            "y_raw": y_raw.detach().cpu().numpy(),
            "y_hat": y_hat.detach().cpu().numpy(),
            "x": x.detach().cpu().numpy()
        })

    def on_validation_epoch_end(self):
        outputs = self.output_cache
        all_y_raw = np.concatenate([x["y_raw"] for x in outputs], axis=0)
        all_y_hat = np.concatenate([x["y_hat"] for x in outputs], axis=0)
        all_x = np.concatenate([x["x"] for x in outputs], axis=0)
        
        scores = metrics(all_y_raw, all_y_hat)
        self.log_dict(scores, sync_dist=False)
        fig = plot_ST(all_x, self.val_pca.transform(all_y_hat))
        self.logger.experiment.add_figure("val/y_hat", fig, self.current_epoch)
        print(scores)

        # Don't forget to clear the memory for the next epoch!
        self.output_cache.clear()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.fitting_model.parameters(), lr=self.pipeline_configs.optimization.lr)
        return optimizer

    def predict_step(self, batch, batch_idx):
        if self.GNN:
            y_raw, x,semantic_neighbors,idx  = batch["raw_representations"], batch["coordinates"],batch["semantic_neighbors"], batch["idx"]
            extra_info = self.extra_variable
            adj = extra_info["adj"]
            raw_rep = extra_info["raw_rep"]
            
            content_neighbours = list(semantic_neighbors.cpu().numpy().flatten()) 
            neighbourhoods = list(set(content_neighbours))
            sub_set_y_raw, sub_set_adj,sub_set_idx = construct_subgraph(raw_rep, adj, neighbourhoods,idx.cpu().numpy())
            sub_set_y_raw = sub_set_y_raw.cuda()
            sub_set_adj = sub_set_adj.cuda()
            y_hat, embd = self.fitting_model.generate(sub_set_y_raw, sub_set_adj, sub_set_idx)

        else:
            y_raw, x = batch["raw_representations"], batch["coordinates"]
            y_hat, embd = self.fitting_model(y_raw)
        self.output_cache.append({
            "x": x.detach().cpu().numpy(),
            "y_raw": y_raw.detach().cpu().numpy(),
            "y_hat": y_hat.detach().cpu().numpy(),
            "embd": embd.detach().cpu().numpy()
        })

    def on_predict_epoch_end(self):
        outputs = self.output_cache
        all_y_raw = np.concatenate([x["y_raw"] for x in outputs], axis=0)
        #all_y_hat = np.concatenate([x["y_hat"] for x in outputs], axis=0)
        all_x = np.concatenate([x["x"] for x in outputs], axis=0)
        all_embd = np.concatenate([x["embd"] for x in outputs], axis=0)
        

        if isinstance(self.pipeline_configs.embedded_data, ListConfig):
            save_name = self.pipeline_configs.embedded_data.pop(0)
        else:
            save_name = self.pipeline_configs.embedded_data

        write_file = os.path.join(self.logger.log_dir, save_name)
        print(f"Writing to {write_file} ... It may take some time ...")
        if save_name.endswith(".h5ad"):
            embedded_data = ad.AnnData(X=all_y_raw)
            embedded_data.obsm["spatial"] = all_x
            embedded_data.obsm["embeddings"] = all_embd
            embedded_data.write(write_file)
        elif save_name.endswith(".npz"):
            np.savez(write_file, spatial=all_x, raw=all_y_raw, embeddings=all_embd)
        


def train_embedder(configs):
    dataset_configs = configs.dataset
    pipeline_configs = configs.pipeline
    
    pl.seed_everything(pipeline_configs.optimization.seed, workers=True) # fix seed globally
    torch.set_float32_matmul_precision("highest") # make use of tensorcore
    if pipeline_configs.embedder.model == 'GAE':
        import scanpy as sc
        from scipy.sparse import issparse
        from sklearn.neighbors import kneighbors_graph
        
        n_neighbors = configs.dataset.n_neighbors
        adata =  sc.read_h5ad(dataset_configs.data_file)
        
        adj = kneighbors_graph(adata.X,n_neighbors,mode='connectivity',n_jobs=8,include_self=True)
        adj = adj.astype(np.float32)
        if dataset_configs.type == 'GraphST2D':
            dataset = GraphST2D(h5ad_file=dataset_configs.data_file, semantic_adj_neighbors=adj.indices.reshape(-1,n_neighbors),  keep_ratio=dataset_configs.keep_ratio)
        elif dataset_configs.type == 'GraphST3D':
            dataset = GraphST3D(h5ad_file=dataset_configs.data_file, semantic_adj_neighbors=adj.indices.reshape(-1,n_neighbors), keep_ratio=dataset_configs.keep_ratio)
        if issparse(adata.X):
            adata.X = adata.X.toarray()
        raw_rep = torch.tensor(adata.X)
        train_idx, val_idx = train_test_split(list(range(len(adata))), test_size=dataset_configs.val_proportion)
        adata_train = adata[train_idx]
        adata_val = adata[val_idx]
        print(f"{val_idx[:10]=}") # check whether seed works
        semantic_adj = kneighbors_graph(adata[train_idx].X,n_neighbors,mode='connectivity',n_jobs=8,include_self=True)
        semantic_adj_val = kneighbors_graph(adata[val_idx].X,n_neighbors,mode='connectivity',n_jobs=8,include_self=True)
        if dataset_configs.type == 'GraphST2D':
            train_dataset = GraphST2D(h5ad_file=adata_train,semantic_adj_neighbors=semantic_adj.indices.reshape(-1,n_neighbors) ,keep_ratio=dataset_configs.keep_ratio)
            val_dataset = GraphST2D(h5ad_file=adata_val,semantic_adj_neighbors=semantic_adj_val.indices.reshape(-1,n_neighbors) , keep_ratio=dataset_configs.keep_ratio)

        elif dataset_configs.type == 'GraphST3D':
            train_dataset = GraphST3D(h5ad_file=adata_train,semantic_adj_neighbors=semantic_adj.indices.reshape(-1,n_neighbors) , keep_ratio=dataset_configs.keep_ratio)
            val_dataset = GraphST3D(h5ad_file=adata_val,semantic_adj_neighbors=semantic_adj_val.indices.reshape(-1,n_neighbors) , keep_ratio=dataset_configs.keep_ratio)
        batch_size = pipeline_configs.optimization.batch_size
        raw_rep_train = torch.tensor(adata[train_idx].X)
        raw_rep_val = torch.tensor(adata[val_idx].X)
        adj_train = semantic_adj.astype(np.float32)
        adj_val = semantic_adj_val.astype(np.float32)

        extra_info = {"raw_rep":raw_rep,"adj": adj,"raw_rep_train": raw_rep_train, "adj_train":adj_train, "raw_rep_val":raw_rep_val, "adj_val":adj_val}
        pipeline_configs.embedder.dim_in = train_dataset.n_gene
        extra_callback = ExtraVariableCallback(extra_variable=extra_info)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=8, drop_last=False)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=8, drop_last=False)

    else:
        # dataset configuration
        dataset_class = getattr(importlib.import_module("datasets"), dataset_configs.type)
        dataset = dataset_class(**dataset_configs)

        pipeline_configs.embedder.dim_in = dataset.get_raw_dim()

        print("Fitting embedder ...")

        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=dataset_configs.val_proportion)
        print(f"{val_idx[:10]=}") # check whether seed works
        train_dataset, val_dataset = Subset(dataset, train_idx), Subset(dataset, val_idx)
        

        batch_size = pipeline_configs.optimization.batch_size
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=8, drop_last=False)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=8, drop_last=False)
        extra_callback = ExtraVariableCallback(extra_variable=None)
    # pipeline configuration
    fitting_system = EmbedderFittingSystem(pipeline_configs, val_pca=dataset.raw_pca)
    tb_logger = pl.loggers.TensorBoardLogger(pipeline_configs.optimization.logs)

    # log raw representations to tensorboard
    train_raw_fig, val_raw_fig = dataset.plot_raw_representations(train_indices=train_idx, val_indices=val_idx)
    tb_logger.experiment.add_figure("dataset/train_raw", train_raw_fig)
    tb_logger.experiment.add_figure("dataset/val_raw", val_raw_fig)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename="{epoch}", save_last=True)

    trainer = L.Trainer(
                #num_sanity_val_steps=0,
                max_epochs=pipeline_configs.optimization.epochs,
                check_val_every_n_epoch=pipeline_configs.optimization.val_freq,
                log_every_n_steps=1, 
                logger=tb_logger, 
                callbacks=[extra_callback, checkpoint_callback],
                devices=1
            )
    trainer.fit(fitting_system, train_dataloader, val_dataloader)

    # predict
    if pipeline_configs.predict_mode=="all":
        test_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=8, drop_last=False)
        trainer.predict(fitting_system, dataloaders=test_dataloader)
    elif pipeline_configs.predict_mode=="val":
        test_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=8, drop_last=False)
        trainer.predict(fitting_system, dataloaders=test_dataloader)
    elif pipeline_configs.predict_mode=="resample":
        test_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=8, drop_last=False)
        trainer.predict(fitting_system, dataloaders=test_dataloader)
        test_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=8, drop_last=False)
        trainer.predict(fitting_system, dataloaders=test_dataloader)
    

    else:
        print("End without writing predictions.")