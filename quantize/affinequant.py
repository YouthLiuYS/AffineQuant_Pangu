import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}

def _adaround_soft(v, zeta, gamma):
    h = torch.sigmoid(v) * (zeta - gamma) + gamma
    return h.clamp(0, 1)

def _adaround_regularizer(module, beta):
    loss = None
    for _, m in module.named_modules():
        if not hasattr(m, "adaround_v") or not hasattr(m, "adaround_enabled"):
            continue
        if not bool(m.adaround_enabled):
            continue
        zeta = getattr(m, "adaround_zeta", 1.1)
        gamma = getattr(m, "adaround_gamma", -0.1)
        h = _adaround_soft(m.adaround_v, zeta, gamma)
        term = (1 - (2 * h - 1).abs().pow(beta)).sum()
        if loss is None:
            loss = term
        else:
            loss = loss + term
    if loss is None:
        return torch.tensor(0.0, device=next(module.parameters()).device)
    return loss

def _adaround_hard_round(module, pos_value=10.0):
    for _, m in module.named_modules():
        if not hasattr(m, "adaround_v") or not hasattr(m, "adaround_enabled"):
            continue
        if not bool(m.adaround_enabled):
            continue
        zeta = getattr(m, "adaround_zeta", 1.1)
        gamma = getattr(m, "adaround_gamma", -0.1)
        h = _adaround_soft(m.adaround_v, zeta, gamma)
        hard = (h >= 0.5).to(dtype=m.adaround_v.dtype)
        m.adaround_v.data = torch.where(
            hard > 0,
            torch.full_like(m.adaround_v, pos_value),
            torch.full_like(m.adaround_v, -pos_value),
        )


def affinequant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    adaround_params=None,
    adaround_layer_idx=-1,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon now")
    
    
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = args.dtype
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    model_dtype = model.dtype
    with torch.no_grad():
        model.to(args.dtype)
    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    with torch.no_grad():
        model.to(model_dtype)
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    
    attention_mask = cache["attention_mask"]
    attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None

    if args.resume:
        affine_parameters = torch.load(args.resume)
    else:
        affine_parameters = {}
    
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        qlayer = DecoderLayer(lm.model.config, layer, args)

        with torch.no_grad():
            qlayer.to(args.dtype)
        # obtain output of full-precision model
        qlayer.set_quant_state(weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]

        # init smooth parameters
        qlayer.set_quant_state(weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        # if is_llama and args.abits == 16:
        #     use_shift = False                   # deactivate channel-wise shifting for llama weight-
        # use_shift = True if args.abits < 16 else False   # only activate per-channel shifting when weight-activation quantization

        use_matrix = args.use_matrix
        use_ln_matrix = args.use_ln_matrix
        if args.let:
            # init channel-wise scaling and shift
            if use_matrix:
                qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.eye(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            else:
                qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=torch.float16).clamp(min=1e-5)
                            weight = module.weight.max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=torch.float16)
                            else:
                                shift = torch.zeros_like(scale)
                            if (pairs[key] == "qkv" or pairs[key] == "fc1") and not use_ln_matrix:
                                qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift.to(args.dtype)))
                                qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale.to(args.dtype)))
                            else:
                                qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift.to(args.dtype)))
                                qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(torch.diag(scale.to(args.dtype))))

        if adaround_params and (adaround_layer_idx < 0 or adaround_layer_idx == i):
            for name, module in qlayer.named_modules():
                if not isinstance(module, QuantLinear):
                    continue
                if hasattr(module, "adaround_enabled"):
                    continue
                key = f"{layer_name_prefix}.{i}.{name}"
                if key not in adaround_params:
                    continue
                v = adaround_params[key].to(device=module.weight.device, dtype=module.weight.dtype)
                module.register_parameter("adaround_v", nn.Parameter(v))
                module.register_buffer(
                    "adaround_zeta",
                    torch.tensor(1.1, dtype=module.weight.dtype, device=module.weight.device),
                )
                module.register_buffer(
                    "adaround_gamma",
                    torch.tensor(-0.1, dtype=module.weight.dtype, device=module.weight.device),
                )
                init_bias = float(v.flatten()[0].item()) if v.numel() > 0 else 0.0
                module.register_buffer(
                    "adaround_init_bias",
                    torch.tensor(init_bias, dtype=module.weight.dtype, device=module.weight.device),
                )
                module.register_buffer(
                    "adaround_enabled",
                    torch.tensor(True, dtype=torch.bool, device=module.weight.device),
                )

        if args.resume and i < len(affine_parameters):
            qlayer.load_state_dict(affine_parameters[i], strict=False)
        
        if args.epochs > 0 and not (args.resume and i < len(affine_parameters)):
            with torch.no_grad():
                qlayer.to(args.dtype)      # required for AMP training
            
            # create optimizer
            if hasattr(qlayer, "adaround_parameters"):
                adaround_layer_params = list(qlayer.adaround_parameters())
            else:
                adaround_layer_params = []
            param_groups = [
                {"params": qlayer.let_parameters(use_shift), "lr": args.let_lr},
                {"params": qlayer.lwc_parameters(), "lr": args.lwc_lr},
            ]
            if adaround_layer_params:
                param_groups.append({"params": adaround_layer_params, "lr": args.let_lr})
            optimizer = torch.optim.AdamW(param_groups, weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            trainable_params = list(qlayer.affine_parameters(use_shift))
            if adaround_layer_params:
                trainable_params.extend(adaround_layer_params)
            
            steps_per_epoch = args.nsamples // args.batch_size
            total_steps = max(steps_per_epoch * max(args.epochs, 1), 1)

            for epochs in range(args.epochs):
                loss_list = []
                recon_loss_list = []
                com_loss_list = []
                norm_list = []

                # gradual mask
                qkvmask_num = int((lm.model.config.hidden_size-1)/(args.epochs-1)*epochs)+1
                fc1mask_num = int((lm.model.config.hidden_size/lm.model.config.num_attention_heads-1)/(args.epochs-1)*epochs)+1
                
                values = torch.tensor([1 for i1 in range(qlayer.self_attn.q_proj.weight.data.size(1))]).cuda()
                maskqkv = torch.zeros(qlayer.self_attn.q_proj.weight.data.size(1), qlayer.self_attn.q_proj.weight.data.size(1)).cuda()
                for i1 in range(qkvmask_num):
                    if i1 == 0:
                        mask1 = torch.diag(values[:len(values)-i1], i1)
                        mask2 = torch.diag(values[:len(values)-i1], -i1)
                    else:
                        mask1 = torch.diag(args.sf*values[:len(values)-i1], i1)
                        mask2 = torch.diag(args.sf*values[:len(values)-i1], -i1)
                    maskqkv = maskqkv + mask1 + mask2
                maskqkv = maskqkv - torch.eye(qlayer.self_attn.q_proj.weight.data.size(1)).cuda()
                
                if "opt" in args.net.lower():
                    maskfc = torch.zeros([qlayer.self_attn.out_proj.weight.data.size(0), qlayer.self_attn.out_proj.weight.data.size(1)]).cuda()
                    head_size = qlayer.self_attn.out_proj.weight.data.size(0)//lm.model.config.num_attention_heads
                elif "llama" in args.net.lower():
                    maskfc = torch.zeros([qlayer.self_attn.o_proj.weight.data.size(0), qlayer.self_attn.o_proj.weight.data.size(1)]).cuda()
                    head_size = qlayer.self_attn.o_proj.weight.data.size(0)//lm.model.config.num_attention_heads
                
                values1 = torch.tensor([1 for i1 in range(head_size)]).cuda()
                ones = torch.zeros(head_size, head_size).cuda()
                for i1 in range(fc1mask_num):
                    if i1 == 0:
                        mask1 = torch.diag(values1[:len(values1)-i1], i1)
                        mask2 = torch.diag(values1[:len(values1)-i1], -i1)
                    else:
                        mask1 = torch.diag(args.sf*values1[:len(values1)-i1], i1)
                        mask2 = torch.diag(args.sf*values1[:len(values1)-i1], -i1)
                    ones = ones + mask1 + mask2
                ones = ones - torch.eye(head_size).cuda()
                for i1 in range(lm.model.config.num_attention_heads):
                    maskfc[i1*head_size:(i1+1)*head_size, i1*head_size:(i1+1)*head_size] = ones
                    
                for j in range(args.nsamples//args.batch_size):  
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast():
                        qlayer.smooth_and_quant_temporary(lm.model.config.num_attention_heads, maskqkv, maskfc, use_matrix=use_matrix, use_ln_matrix=use_ln_matrix)
                        quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                        recon_loss = loss
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)
                        if adaround_layer_params and args.adaround_reg_weight > 0:
                            if args.epochs > 1:
                                progress = epochs / (args.epochs - 1)
                            else:
                                progress = 1.0
                            beta = args.adaround_beta_start + (args.adaround_beta_end - args.adaround_beta_start) * progress
                            loss_com = _adaround_regularizer(qlayer, beta)
                            scale = recon_loss.detach() / (loss_com.detach() + 1e-12)
                            loss_com = loss_com * scale
                            loss = loss + args.adaround_reg_weight * loss_com
                            com_loss = loss_com
                        else:
                            com_loss = torch.tensor(0.0, device=loss.device)
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.data)
                    recon_loss_list.append(recon_loss.data)
                    com_loss_list.append(com_loss.data)
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer, parameters=trainable_params)
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                recon_mean = torch.stack(recon_loss_list).mean()
                com_mean = torch.stack(com_loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(
                    f"layer {i} iter {epochs} loss:{loss_mean} recon:{recon_mean} L_com:{com_mean} "
                    f"norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} "
                )

            if adaround_layer_params:
                _adaround_hard_round(qlayer)
            qlayer.clear_temp_variable()
            del optimizer

        if args.resume and i < len(affine_parameters):
            qkvmask_num = lm.model.config.hidden_size
            fc1mask_num = lm.model.config.hidden_size//lm.model.config.num_attention_heads
            values = torch.tensor([1 for i1 in range(qlayer.self_attn.q_proj.weight.data.size(1))]).cuda()
            maskqkv = torch.zeros(qlayer.self_attn.q_proj.weight.data.size(1), qlayer.self_attn.q_proj.weight.data.size(1)).cuda()
            for i1 in range(qkvmask_num):
                if i1 == 0:
                    mask1 = torch.diag(values[:len(values)-i1], i1)
                    mask2 = torch.diag(values[:len(values)-i1], -i1)
                else:
                    mask1 = torch.diag(args.sf*values[:len(values)-i1], i1)
                    mask2 = torch.diag(args.sf*values[:len(values)-i1], -i1)
                maskqkv = maskqkv + mask1 + mask2
            maskqkv = maskqkv - torch.eye(qlayer.self_attn.q_proj.weight.data.size(1)).cuda()

            if "opt" in args.net.lower():
                maskfc = torch.zeros([qlayer.self_attn.out_proj.weight.data.size(0), qlayer.self_attn.out_proj.weight.data.size(1)]).cuda()
                head_size = qlayer.self_attn.out_proj.weight.data.size(0)//lm.model.config.num_attention_heads
            elif "llama" in args.net.lower():
                maskfc = torch.zeros([qlayer.self_attn.o_proj.weight.data.size(0), qlayer.self_attn.o_proj.weight.data.size(1)]).cuda()
                head_size = qlayer.self_attn.o_proj.weight.data.size(0)//lm.model.config.num_attention_heads
            
            values1 = torch.tensor([1 for i1 in range(head_size)]).cuda()
            ones = torch.zeros(head_size, head_size).cuda()
            for i1 in range(fc1mask_num):
                if i1 == 0:
                    mask1 = torch.diag(values1[:len(values1)-i1], i1)
                    mask2 = torch.diag(values1[:len(values1)-i1], -i1)
                else:
                    mask1 = torch.diag(args.sf*values1[:len(values1)-i1], i1)
                    mask2 = torch.diag(args.sf*values1[:len(values1)-i1], -i1)
                ones = ones + mask1 + mask2
            ones = ones - torch.eye(head_size).cuda()
            for i1 in range(lm.model.config.num_attention_heads):
                maskfc[i1*head_size:(i1+1)*head_size, i1*head_size:(i1+1)*head_size] = ones


        # real smooth and quantization
        qlayer.smooth_and_quant_inplace(lm.model.config.num_attention_heads, maskqkv, maskfc,use_matrix=use_matrix,use_ln_matrix=use_ln_matrix)
        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
            qlayer.register_scales_and_zeros()
            layers[i] = qlayer.to("cpu")
            affine_parameters[i] = qlayer.affine_state_dict()
            torch.save(affine_parameters, os.path.join(args.output_dir, "affine_parameters.pth"))
        else:
            qlayer.register_scales_and_zeros()
            qlayer.half()
            layers[i] = qlayer.to("cpu")
        if args.real_quant:
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.float().cpu(),  scales.float().cpu(), zeros.float().cpu())
                
                levels = name.split('.')
                if len(levels) > 1:
                    mod_ = qlayer
                    for l_idx in range(len(levels)-1):
                        if levels[l_idx].isdigit():
                            mod_ = mod_[int(levels[l_idx])]
                        else:
                            mod_ = getattr(mod_, levels[l_idx])
                    setattr(mod_, levels[-1], q_linear)
                else:
                    setattr(qlayer, name, q_linear)        
                del module        

        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model.half()
