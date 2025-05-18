import argparse
from omegaconf import OmegaConf
from utils import pprint_config
from systems import predict_inr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['inr'], required=True)
    parser.add_argument('--conf', type=str, required=True)
    args = parser.parse_args()
    configs = OmegaConf.load(args.conf)
    print(args.conf)
    pprint_config(configs)

    if args.mode == "inr":
        predict_inr(configs)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
