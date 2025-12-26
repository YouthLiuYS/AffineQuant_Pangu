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
import wandb # Import wandb



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def affinequant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
    use_wandb=False, # Add use_wandb parameter
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
    
    global_step = 0 # Initialize global step counter
    # Dictionary to collect loss data for each layer: {layer_id: [(epoch, loss), ...]}
    layer_loss_data = {}
    # List to collect parameter data for bar chart: [(layer_index, scale_params, shift_params, total_trainable_params), ...]
    all_layers_param_data = []
    
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        # Set current layer name for group_size configuration
        args.current_layer_name = f"{layer_name_prefix}.{i}"
        qlayer = DecoderLayer(lm.model.config, layer, args)

        # --- Calculate Initial Spectral Norms ---
        init_sn = {}
        with torch.no_grad():
            for n, m in qlayer.named_modules():
                if isinstance(m, QuantLinear):
                    try:
                        # Spectral norm is the largest singular value (2-norm)
                        sn = torch.linalg.norm(m.weight.float(), ord=2).item()
                        init_sn[n] = sn
                    except Exception as e:
                        logger.warning(f"Failed to calc spectral norm for {n}: {e}")
            logger.info(f"Layer {i} Initial Spectral Norms: {init_sn}")
            if use_wandb:
                wandb.log({f"spectral_norm_init/layer_{i}/{k}": v for k, v in init_sn.items()})
                if init_sn:
                    wandb.log({f"spectral_norm_init/layer_{i}_max": max(init_sn.values())})
        # ----------------------------------------

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

        if args.resume and i < len(affine_parameters):
            qlayer.load_state_dict(affine_parameters[i], strict=False)
        
        if args.epochs > 0 and not (args.resume and i < len(affine_parameters)):
            with torch.no_grad():
                qlayer.to(args.dtype)      # required for AMP training
            
            # create optimizer
            optimizer = torch.optim.AdamW(
                [{"params":qlayer.let_parameters(use_shift),"lr":args.let_lr}, {"params":qlayer.lwc_parameters(),"lr":args.lwc_lr}],weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            for epochs in range(args.epochs):
                loss_list = []
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
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.data)
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters=qlayer.affine_parameters(use_shift))
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
                
                if use_wandb:
                    # Collect loss data for this layer
                    if i not in layer_loss_data:
                        layer_loss_data[i] = []
                    layer_loss_data[i].append((epochs, loss_mean.item()))
                    
                    # Log individual layer loss with global step for real-time monitoring
                    global_step += 1
                    wandb.log({
                        f"loss/layer_{i}": loss_mean.item(),
                        "layer": i,
                        "epoch": epochs,
                        "global_step": global_step
                    })
            
            # 打印当前层的参数量
            layer_params = sum(p.numel() for p in qlayer.parameters())
            layer_trainable_params = sum(p.numel() for p in qlayer.parameters() if p.requires_grad)
            
            # 统计当前层的 scale 和 shift 参数量
            scale_params = sum(p.numel() for name, p in qlayer.named_parameters() if 'scale' in name.lower())
            shift_params = sum(p.numel() for name, p in qlayer.named_parameters() if 'shift' in name.lower())
            logger.info(f"Layer {i} params: total={layer_params:,}, trainable={layer_trainable_params:,}, scale={scale_params:,}, shift={shift_params:,}")

            if use_wandb:
                # Collect parameter data for final plot
                all_layers_param_data.append((i, scale_params, shift_params, layer_trainable_params))
                
                # Also log immediately for real-time view
                wandb.log({
                    f"params/layer_{i}_scale": scale_params,
                    f"params/layer_{i}_shift": shift_params,
                    f"params/layer_{i}_trainable": layer_trainable_params,
                    "layer": i
                })
                
                # Update loss curves plot in real-time after each layer completes
                try:
                    import matplotlib.pyplot as plt
                    import matplotlib.cm as cm
                    import numpy as np
                    
                    fig, ax = plt.subplots(figsize=(18, 6))
                    
                    # Use a colormap to assign different colors to each layer
                    num_completed_layers = len(layer_loss_data)
                    colors = cm.tab20(np.linspace(0, 1, num_completed_layers)) if num_completed_layers <= 20 else cm.viridis(np.linspace(0, 1, num_completed_layers))
                    
                    # Track global step counter
                    current_global_step = 0
                    layer_boundaries = []
                    
                    for idx, (layer_id, loss_list) in enumerate(sorted(layer_loss_data.items())):
                        # Assign global steps for this layer
                        global_steps = []
                        losses = []
                        
                        for epoch, loss in loss_list:
                            global_steps.append(current_global_step)
                            losses.append(loss)
                            current_global_step += 1
                        
                        # Plot this layer's curve (points within same layer are connected)
                        ax.plot(global_steps, losses, marker='o', markersize=4, linewidth=1.5, 
                               color=colors[idx], label=f'Layer {layer_id}', alpha=0.8)
                        
                        # Record layer boundary for visualization
                        if global_steps:
                            layer_boundaries.append((global_steps[0], layer_id))
                    
                    # Add vertical lines at layer boundaries (except the first one)
                    for idx in range(1, len(layer_boundaries)):
                        boundary_x, layer_id = layer_boundaries[idx]
                        ax.axvline(x=boundary_x - 0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
                    
                    ax.set_xlabel('Global Step', fontsize=12)
                    ax.set_ylabel('Loss', fontsize=12)
                    ax.set_title(f'Loss Curves by Layer (逐层量化Loss变化) - {num_completed_layers} layers completed', fontsize=14)
                    ax.grid(True, alpha=0.3)
                    
                    # Create legend
                    if num_completed_layers > 10:
                        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=max(1, num_completed_layers // 15))
                        plt.tight_layout(rect=[0, 0, 0.88, 1])
                    else:
                        ax.legend(loc='best', fontsize=9)
                        plt.tight_layout()
                    
                    # Log the matplotlib figure to wandb (will update in real-time)
                    wandb.log({"Loss_Curves_All_Layers": wandb.Image(fig)})
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"Failed to update real-time loss curves plot: {e}")

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

        # --- Calculate Final Spectral Norms ---
        final_sn = {}
        with torch.no_grad():
            for n, m in qlayer.named_modules():
                if isinstance(m, QuantLinear):
                    try:
                        sn = torch.linalg.norm(m.weight.float(), ord=2).item()
                        final_sn[n] = sn
                    except Exception as e:
                        logger.warning(f"Failed to calc final spectral norm for {n}: {e}")
            logger.info(f"Layer {i} Final Spectral Norms: {final_sn}")
            if use_wandb:
                wandb.log({f"spectral_norm_final/layer_{i}/{k}": v for k, v in final_sn.items()})
                if final_sn:
                    wandb.log({f"spectral_norm_final/layer_{i}_max": max(final_sn.values())})
                
                # Log the ratio (Growth/Change)
                for k, v in final_sn.items():
                    if k in init_sn and init_sn[k] > 0:
                        wandb.log({f"spectral_norm_ratio/layer_{i}/{k}": v / init_sn[k]})
        # --------------------------------------

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

    if use_wandb and layer_loss_data:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import numpy as np
            
            # Create a simple plot: x=global_step, y=loss
            # Each layer is a separate line segment (not connected to other layers)
            fig, ax = plt.subplots(figsize=(18, 6))
            
            # Use a colormap to assign different colors to each layer
            num_layers = len(layer_loss_data)
            colors = cm.tab20(np.linspace(0, 1, num_layers)) if num_layers <= 20 else cm.viridis(np.linspace(0, 1, num_layers))
            
            # Track global step counter
            current_global_step = 0
            layer_boundaries = []
            
            for idx, (layer_id, loss_list) in enumerate(sorted(layer_loss_data.items())):
                # Assign global steps for this layer
                global_steps = []
                losses = []
                
                for epoch, loss in loss_list:
                    global_steps.append(current_global_step)
                    losses.append(loss)
                    current_global_step += 1
                
                # Plot this layer's curve (points within same layer are connected)
                ax.plot(global_steps, losses, marker='o', markersize=4, linewidth=1.5, 
                       color=colors[idx], label=f'Layer {layer_id}', alpha=0.8)
                
                # Record layer boundary for visualization
                if global_steps:
                    layer_boundaries.append((global_steps[0], layer_id))
            
            # Add vertical lines at layer boundaries (except the first one)
            for i in range(1, len(layer_boundaries)):
                boundary_x, layer_id = layer_boundaries[i]
                ax.axvline(x=boundary_x - 0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            
            ax.set_xlabel('Global Step', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Loss Curves by Layer (逐层量化Loss变化)', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Create legend
            if num_layers > 10:
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=max(1, num_layers // 15))
                plt.tight_layout(rect=[0, 0, 0.88, 1])
            else:
                ax.legend(loc='best', fontsize=9)
                plt.tight_layout()
            
            # Log the matplotlib figure to wandb
            wandb.log({"Loss_Curves_All_Layers": wandb.Image(fig)})
            plt.close(fig)
            
            # Also create a line series chart for interactive viewing (original overlapped view)
            wandb.log({
                "Loss_Curves_by_Layer_Interactive": wandb.plot.line_series(
                    xs=[list(range(len(loss_list))) for loss_list in layer_loss_data.values()],
                    ys=[[loss for _, loss in loss_list] for loss_list in layer_loss_data.values()],
                    keys=[f"Layer_{layer_id}" for layer_id in layer_loss_data.keys()],
                    title="Loss Curves by Layer (Interactive - Overlapped)",
                    xname="Epoch"
                )
            })
            logger.info("Logged Loss Curves plots to WandB.")
        except Exception as e:
            logger.error(f"Failed to log Loss Curves plot to WandB: {e}")

    if use_wandb and all_layers_param_data:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create parameter counts plot with three lines: scale, shift, total_trainable
            layer_indices = [item[0] for item in all_layers_param_data]
            scale_params_list = [item[1] for item in all_layers_param_data]
            shift_params_list = [item[2] for item in all_layers_param_data]
            trainable_params_list = [item[3] for item in all_layers_param_data]
            
            # Create matplotlib figure for parameter counts
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.array(layer_indices)
            width = 0.25
            
            bars1 = ax.bar(x - width, scale_params_list, width, label='Scale参数量', color='#2ecc71', alpha=0.8)
            bars2 = ax.bar(x, shift_params_list, width, label='Shift参数量', color='#3498db', alpha=0.8)
            bars3 = ax.bar(x + width, trainable_params_list, width, label='总可训练参数量', color='#e74c3c', alpha=0.8)
            
            ax.set_xlabel('Layer Index', fontsize=12)
            ax.set_ylabel('Parameter Count', fontsize=12)
            ax.set_title('Parameter Counts by Layer (各层参数量统计)', fontsize=14)
            ax.set_xticks(x)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Log the matplotlib figure to wandb
            wandb.log({"Param_Counts_All_Layers": wandb.Image(fig)})
            plt.close(fig)
            
            # Also create a line chart for interactive viewing
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(layer_indices, scale_params_list, marker='o', linewidth=2, label='Scale参数量', color='#2ecc71')
            ax2.plot(layer_indices, shift_params_list, marker='s', linewidth=2, label='Shift参数量', color='#3498db')
            ax2.plot(layer_indices, trainable_params_list, marker='^', linewidth=2, label='总可训练参数量', color='#e74c3c')
            
            ax2.set_xlabel('Layer Index', fontsize=12)
            ax2.set_ylabel('Parameter Count', fontsize=12)
            ax2.set_title('Parameter Counts by Layer (Line Chart)', fontsize=14)
            ax2.legend(loc='upper right', fontsize=10)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            
            wandb.log({"Param_Counts_Line_Chart": wandb.Image(fig2)})
            plt.close(fig2)
            
            # Log line series chart for interactive viewing in wandb
            wandb.log({
                "Param_Counts_by_Layer_Interactive": wandb.plot.line_series(
                    xs=[layer_indices, layer_indices, layer_indices],
                    ys=[scale_params_list, shift_params_list, trainable_params_list],
                    keys=["Scale参数量", "Shift参数量", "总可训练参数量"],
                    title="Parameter Counts by Layer (Interactive)",
                    xname="Layer Index"
                )
            })
            logger.info("Logged Parameter Counts plots to WandB.")
        except Exception as e:
            logger.error(f"Failed to log Parameter Counts plot to WandB: {e}")

    return model.half()

