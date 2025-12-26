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


def get_adaround_params(model, adaround_mode="full", init_bias=-5.0):
    adaround = {}
    try:
        from quantize.int_linear import QuantLinear
    except Exception:
        QuantLinear = None

    has_quant_linear = False
    if QuantLinear is not None:
        for m in model.modules():
            if isinstance(m, QuantLinear):
                has_quant_linear = True
                break

    mode = (adaround_mode or "full").lower()
    for name, m in model.named_modules():
        if has_quant_linear:
            if QuantLinear is None or not isinstance(m, QuantLinear):
                continue
        else:
            if not isinstance(m, nn.Linear):
                continue

        if mode in ("full", "full_matrix"):
            v_shape = m.weight.shape
        elif mode in ("per_channel", "channel", "out_channel"):
            v_shape = (m.weight.shape[0], 1)
        else:
            raise ValueError(f"Unsupported adaround_mode: {adaround_mode}")

        adaround[name] = torch.full(
            v_shape,
            float(init_bias),
            dtype=m.weight.dtype,
            device="cpu",
        )

    return adaround


def get_adaround(
    model,
    adaround_mode="full",
    init_bias=-5.0,
    zeta=1.1,
    gamma=-0.1,
):
    adaround_params = []
    try:
        from quantize.int_linear import QuantLinear
    except Exception:
        QuantLinear = None

    has_quant_linear = False
    if QuantLinear is not None:
        for m in model.modules():
            if isinstance(m, QuantLinear):
                has_quant_linear = True
                break

    mode = (adaround_mode or "full").lower()
    for _, m in model.named_modules():
        if has_quant_linear:
            if QuantLinear is None or not isinstance(m, QuantLinear):
                continue
        else:
            if not isinstance(m, nn.Linear):
                continue
        if hasattr(m, "adaround_enabled"):
            continue

        weight = m.weight
        if mode in ("full", "full_matrix"):
            v_shape = weight.shape
        elif mode in ("per_channel", "channel", "out_channel"):
            v_shape = (weight.shape[0], 1)
        else:
            raise ValueError(f"Unsupported adaround_mode: {adaround_mode}")

        v_param = nn.Parameter(
            torch.full(
                v_shape,
                float(init_bias),
                dtype=weight.dtype,
                device=weight.device,
            )
        )
        m.register_parameter("adaround_v", v_param)
        m.register_buffer(
            "adaround_zeta",
            torch.tensor(float(zeta), dtype=weight.dtype, device=weight.device),
        )
        m.register_buffer(
            "adaround_gamma",
            torch.tensor(float(gamma), dtype=weight.dtype, device=weight.device),
        )
        m.register_buffer(
            "adaround_init_bias",
            torch.tensor(float(init_bias), dtype=weight.dtype, device=weight.device),
        )
        m.register_buffer(
            "adaround_enabled",
            torch.tensor(True, dtype=torch.bool, device=weight.device),
        )
        adaround_params.append(v_param)

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
    parser.add_argument('--adaround-output-path', type=str, default='./adaround_params/',
                        help='where to save the adaround params')
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix","pile"],
        help="Where to extract calibration data from.",)
    parser.add_argument('--num-samples', type=int, default=128)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--adaround-mode', type=str, default="full",
                        choices=["full", "full_matrix", "per_channel", "channel", "out_channel"])
    parser.add_argument('--adaround-init-bias', type=float, default=-5.0)
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
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

    adaround_params = get_adaround_params(
        model,
        adaround_mode=args.adaround_mode,
        init_bias=args.adaround_init_bias,
    )
    save_path = os.path.join(args.adaround_output_path, f'{args.net}.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(adaround_params, save_path)


if __name__ == '__main__':
    main()
