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



def _compute_adaround_v(
    weight,
    quantizer,
    mode,
    init_bias,
    zeta,
    gamma,
    weight_quant_params=None,
):
    def _frac_from_weight(w, scale, group_size=None, deficiency=0):
        if group_size:
            dim1, dim2 = w.shape
            if deficiency > 0:
                pad_zeros = torch.zeros((dim1, deficiency), dtype=w.dtype, device=w.device)
                w = torch.cat((w, pad_zeros), dim=1)
            dim1_p, dim2_p = w.shape
            w = w.reshape(-1, group_size)
            frac = w / scale
            frac = frac - torch.floor(frac)
            frac = frac.reshape(dim1_p, dim2_p)
            if deficiency > 0:
                frac = frac[:, :dim2]
        else:
            frac = w / scale
            frac = frac - torch.floor(frac)
        return frac

    if quantizer is not None and hasattr(quantizer, "per_token_dynamic_calibration"):
        with torch.no_grad():
            w = weight.detach().float()
            quantizer.per_token_dynamic_calibration(w)
            scale = quantizer.scale
            if scale is None:
                return torch.full_like(weight, float(init_bias))
            scale = scale.float()
            group_size = getattr(quantizer, "group_size", None)
            deficiency = getattr(quantizer, "deficiency", 0)
            frac = _frac_from_weight(w, scale, group_size, deficiency)
    elif weight_quant_params is not None:
        n_bits = weight_quant_params.get("n_bits", 8)
        symmetric = weight_quant_params.get("symmetric", False)
        group_size = weight_quant_params.get("group_size", None)
        lwc = weight_quant_params.get("lwc", False)

        with torch.no_grad():
            w = weight.detach().float()
            deficiency = 0
            if group_size:
                remainder = w.shape[1] % group_size
                if remainder > 0:
                    deficiency = group_size - remainder
                if deficiency > 0:
                    pad_zeros = torch.zeros((w.shape[0], deficiency), dtype=w.dtype, device=w.device)
                    w = torch.cat((w, pad_zeros), dim=1)
                w = w.reshape(-1, group_size)
            xmin = w.amin(dim=-1, keepdim=True)
            xmax = w.amax(dim=-1, keepdim=True)
            if lwc:
                factor = torch.sigmoid(torch.tensor(4.0, dtype=w.dtype, device=w.device))
                xmax = factor * xmax
                xmin = factor * xmin
            if symmetric:
                abs_max = torch.max(xmax.abs(), xmin.abs())
                scale = abs_max / (2**(n_bits - 1) - 1)
            else:
                scale = (xmax - xmin) / (2**n_bits - 1)
            scale = scale.clamp(min=1e-5, max=1e4)

            if group_size:
                dim1_p = weight.shape[0]
                dim2_p = weight.shape[1] + deficiency
                w = weight.detach().float()
                if deficiency > 0:
                    pad_zeros = torch.zeros((dim1_p, deficiency), dtype=w.dtype, device=w.device)
                    w = torch.cat((w, pad_zeros), dim=1)
                w = w.reshape(-1, group_size)
                frac = w / scale
                frac = frac - torch.floor(frac)
                frac = frac.reshape(dim1_p, dim2_p)
                if deficiency > 0:
                    frac = frac[:, :weight.shape[1]]
            else:
                w = weight.detach().float()
                frac = w / scale
                frac = frac - torch.floor(frac)
    else:
        return torch.full_like(weight, float(init_bias))

    frac = frac.clamp(0.0, 1.0)
    if mode in ("per_channel", "channel", "out_channel"):
        h = frac.mean(dim=1, keepdim=True)
    else:
        h = frac
    p = (h - gamma) / (zeta - gamma)
    p = p.clamp(1e-6, 1 - 1e-6)
    v = torch.log(p) - torch.log1p(-p)
    return v.to(dtype=weight.dtype)


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


def get_adaround_params(
    model,
    adaround_mode="full",
    init_bias=-5.0,
    zeta=1.1,
    gamma=-0.1,
    weight_quant_params=None,
):
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

        v = _compute_adaround_v(
            m.weight,
            getattr(m, "weight_quantizer", None),
            mode,
            init_bias,
            zeta,
            gamma,
            weight_quant_params=weight_quant_params,
        )
        if mode in ("per_channel", "channel", "out_channel") and v.shape != (m.weight.shape[0], 1):
            v = v.mean(dim=1, keepdim=True)
        adaround[name] = v.to(device="cpu", dtype=m.weight.dtype)

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
        if mode in ("full", "full_matrix", "per_channel", "channel", "out_channel"):
            v = _compute_adaround_v(
                weight,
                getattr(m, "weight_quantizer", None),
                mode,
                init_bias,
                zeta,
                gamma,
            )
            if mode in ("per_channel", "channel", "out_channel") and v.shape != (weight.shape[0], 1):
                v = v.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unsupported adaround_mode: {adaround_mode}")

        v_param = nn.Parameter(v.to(device=weight.device, dtype=weight.dtype))
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
    parser.add_argument('--wbits', type=int, default=4)
    parser.add_argument('--symmetric', default=False, action="store_true")
    parser.add_argument('--group_size', type=int, default=None)
    parser.add_argument('--lwc', default=False, action="store_true")
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

    weight_quant_params = {
        "n_bits": args.wbits,
        "symmetric": args.symmetric,
        "group_size": args.group_size,
        "lwc": args.lwc,
    }
    adaround_params = get_adaround_params(
        model,
        adaround_mode=args.adaround_mode,
        init_bias=args.adaround_init_bias,
        weight_quant_params=weight_quant_params,
    )
    save_path = os.path.join(args.adaround_output_path, f'{args.net}.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(adaround_params, save_path)


if __name__ == '__main__':
    main()
