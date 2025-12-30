import os
import torch


def print_adaround_params(path="adaround_params/opt-125m.pt"):
    """
    Load and print the contents of an AdaRound params file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"AdaRound params file not found: {path}")
    params = torch.load(path, weights_only=False)
    print(params)


def print_adaround_params_log(path="/home/lys/pangu/AffineQuant/log/opt-125m-w4a16/adaround_params.pth"):
    """
    Load and print the contents of the AdaRound params log file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"AdaRound params file not found: {path}")
    params = torch.load(path, weights_only=False)
    print(params)
