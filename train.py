import argparse
from omegaconf import OmegaConf
from utils import pprint_config
from systems import train_embedder, train_inr, fit_griddata, fit_pca


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['embedder', 'inr', 'grid', 'pca'], required=True)
    parser.add_argument('--conf', type=str, required=True)
    args = parser.parse_args()
    configs = OmegaConf.load(args.conf)

    pprint_config(configs)

    if args.mode == "embedder":
        train_embedder(configs)
    elif args.mode == "inr":
        train_inr(configs)
    elif args.mode == "grid":
        fit_griddata(configs)
    elif args.mode == "pca":
        fit_pca(configs)

    pprint_config(configs)