import torch
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)
import argparse
import torch.nn as nn

from datasets import load_dataset
import functools
from tqdm import tqdm
from datautils import get_loaders

try:
    from quantize.int_linear import QuantLinear
except ImportError:
    QuantLinear = None
# try:
#     from llava.model import *   # required for llava
# except ImportError:
#     print("If want to quantize llave models, you should manually install llava from https://github.com/haotian-liu/LLaVA")

# import pdb



def get_act_scales(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))

    for h in hooks:
        h.remove()

    return act_scales

def get_act_shifts(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
    act_shifts = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        comming_min = torch.min(tensor, dim=0)[0].float().cpu()
        if name in act_shifts:
            act_shifts[name] = 0.99*act_shifts[name] + 0.01 *((comming_max+comming_min)/2)
        else:
            act_shifts[name] = (comming_max+comming_min)/2

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))


    for h in hooks:
        h.remove()

    return act_shifts


def get_adaround(model, init_bias=0.0, zeta=1.1, gamma=-0.1):
    """
    Initialize AdaRound parameters (V matrix) for linear layers.
    For AutoModelForCausalLM (nn.Linear), generates parameter dict only.
    Returns: dict[name -> Tensor] for all linear layer weights.
    """
    adaround_params = {}

    for name, m in model.named_modules():
        # Prefer QuantLinear if available, fallback to nn.Linear
        if QuantLinear is not None and isinstance(m, QuantLinear):
            weight = m.weight
        elif isinstance(m, nn.Linear):
            weight = m.weight
        else:
            continue

        # Initialize V with constant init_bias (negative value)
        # V shape matches weight shape: (out_features, in_features)
        V = torch.full_like(weight, init_bias, dtype=torch.float32)
        adaround_params[name] = V.cpu()

    return adaround_params


def build_model_and_tokenizer(model_name):
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='./llama-7b', help='model name')
    parser.add_argument('--scales-output-path', type=str, default='./act_scales/',
                        help='where to save the act scales')
    parser.add_argument('--shifts-output-path', type=str, default='./act_shifts/',
                        help='where to save the act shifts')
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix","pile"],
        help="Where to extract calibration data from.",)
    parser.add_argument('--num-samples', type=int, default=128)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    # AdaRound parameters
    parser.add_argument('--adaround-output-path', type=str, default='./adaround_params/',
                        help='where to save the adaround parameters')
    parser.add_argument('--adaround-init-bias', type=float, default=0.0,
                        help='initial bias for adaround V matrix (0.0 gives h=0.5, i.e. round-to-nearest)')
    parser.add_argument('--adaround-zeta', type=float, default=1.1,
                        help='zeta parameter for adaround rectified sigmoid')
    parser.add_argument('--adaround-gamma', type=float, default=-0.1,
                        help='gamma parameter for adaround rectified sigmoid')
    parser.add_argument('--generate-adaround', action='store_true',
                        help='generate adaround parameters')
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model)
    dataloader, _ = get_loaders(
    args.calib_dataset,
    nsamples=args.num_samples,
    seed=args.seed,
    model=args.model,
    seqlen=args.seq_len,
    )

    args.net = args.model.split('/')[-1]
    act_scales = get_act_scales(model, dataloader,args.num_samples)
    save_path = os.path.join(args.scales_output_path,f'{args.net}.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_scales, save_path)

    act_shifts = get_act_shifts(model, dataloader,args.num_samples)
    save_path = os.path.join(args.shifts_output_path,f'{args.net}.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_shifts, save_path)

    # Generate and save AdaRound parameters if requested
    if args.generate_adaround:
        adaround_params = get_adaround(
            model,
            init_bias=args.adaround_init_bias,
            zeta=args.adaround_zeta,
            gamma=args.adaround_gamma
        )
        save_path = os.path.join(args.adaround_output_path, f'{args.net}.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(adaround_params, save_path)
        print(f"AdaRound parameters saved to {save_path}")


if __name__ == '__main__':
    main()
