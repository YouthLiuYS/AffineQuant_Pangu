import torch
# from torch._six import inf
from math import inf
import logging
from termcolor import colored
import sys
import os
import time


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True,retain_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}_{int(time.time())}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger

def get_group_size(config, module_name, default_group_size):
    """
    Get group size for a specific module based on configuration.
    
    Args:
        config (dict): Dictionary containing group size configuration.
        module_name (str): Full name of the module (e.g. "model.decoder.layers.0.self_attn.q_proj").
        default_group_size (int): Default group size to use if not specified in config.
        
    Returns:
        int: The group size to use.
    """
    if not config:
        return default_group_size
        
    # Check for exact match
    if module_name in config:
        return config[module_name]
    
    # Try to extract layer index and check for integer key match
    try:
        # Example: "model.decoder.layers.0.self_attn.q_proj" -> "0"
        # Example: "model.layers.1.mlp.gate_proj" -> "1"
        parts = module_name.split('.')
        # Find the part that looks like "layers.<index>"
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i+1].isdigit():
                layer_index = int(parts[i+1])
                if layer_index in config:
                    return config[layer_index]
    except Exception:
        # If any parsing fails, just continue to other checks
        pass
        
    # Check for partial matches (keys in config that are substrings of module_name)
    # We prioritize longer matches (more specific)
    matched_key = None
    max_len = -1
    
    for key in config:
        # Ensure key is a string for 'in' operator
        if isinstance(key, str) and key in module_name:
            if len(key) > max_len:
                max_len = len(key)
                matched_key = key
                
    if matched_key is not None:
        return config[matched_key]
        
    return default_group_size