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


def adaround_reg_loss(V, zeta, gamma, beta):
    """
    Compute AdaRound regularization loss L_com.
    L_com = sum(1 - |2 * h(V) - 1|^beta)
    where h(V) = clamp(sigmoid(V) * (zeta - gamma) + gamma, 0, 1)
    """
    h = torch.clamp(torch.sigmoid(V) * (zeta - gamma) + gamma, 0, 1)
    reg = 1 - torch.pow((2 * h - 1).abs(), beta)
    return reg.sum()


def get_adaround_h(V, zeta, gamma, hard=False):
    """
    Compute AdaRound soft/hard rounding offset h(V).
    h = clamp(sigmoid(V) * (zeta - gamma) + gamma, 0, 1)
    If hard=True, return (h >= 0.5).float()
    """
    h = torch.clamp(torch.sigmoid(V) * (zeta - gamma) + gamma, 0, 1)
    if hard:
        h = (h >= 0.5).float()
    return h


def compute_adaround_v_init(smooth_weight, quantizer, zeta, gamma, dev, dtype):
    """
    Compute V initialization based on smooth weights' round-to-nearest.

    Steps:
    1. Compute scale/zero using quantizer logic (with group_size/deficiency padding)
    2. x_scaled = w_smooth / scale
    3. h0 = round(x_scaled) - floor(x_scaled)  # 0 or 1
    4. p = clamp((h0 - gamma) / (zeta - gamma), eps, 1-eps)
    5. V_init = log(p / (1 - p))  # inverse sigmoid
    """
    w = smooth_weight.clone()
    deficiency = quantizer.deficiency
    group_size = quantizer.group_size

    # Pad if needed (same logic as fake_quant)
    if deficiency > 0:
        pad_zeros = torch.zeros((w.shape[0], deficiency), dtype=w.dtype, device=w.device)
        w = torch.cat((w, pad_zeros), dim=1)

    # Reshape for group quantization
    if group_size:
        dim1, dim2 = w.shape
        w = w.reshape(-1, group_size)

    # Compute scale and zero_point using quantizer's calibration
    quantizer.per_token_dynamic_calibration(w)
    scale = quantizer.scale

    # Compute x_scaled = w / scale
    x_scaled = w / scale

    # h0 = round(x_scaled) - floor(x_scaled)
    x_round = torch.round(x_scaled)
    x_floor = torch.floor(x_scaled)
    h0 = x_round - x_floor  # Should be 0 or 1

    # Clamp h0 to [0, 1] for safety
    h0 = torch.clamp(h0, 0, 1)

    # Compute V_init using inverse of rectified sigmoid
    # h = clamp(sigmoid(V) * (zeta - gamma) + gamma, 0, 1)
    # Solving for V: p = (h0 - gamma) / (zeta - gamma), V = log(p / (1-p))
    eps = 1e-6
    p = (h0 - gamma) / (zeta - gamma)
    p = torch.clamp(p, eps, 1 - eps)
    V_init = torch.log(p / (1 - p))

    # Reshape back
    if group_size:
        V_init = V_init.reshape(dim1, dim2)
    if deficiency > 0:
        V_init = V_init[:, :-deficiency]

    return V_init.to(device=dev, dtype=dtype)


def init_adaround_params_for_module(module, name, adaround_params, init_bias, zeta, gamma, dev, dtype, use_smooth_init=False):
    """
    Initialize AdaRound parameters for a QuantLinear module.

    If use_smooth_init=True and module has smooth_weight, initialize V based on
    smooth weight's round-to-nearest. Otherwise use init_bias or loaded params.

    IMPORTANT: smooth_weight is kept for AdaRound training (not deleted here).

    Returns the V parameter tensor.
    """
    if hasattr(module, 'adaround_enabled') and module.adaround_enabled:
        return module.adaround_V

    weight = module.weight

    # Determine V initialization
    if use_smooth_init and hasattr(module, 'smooth_weight'):
        # Initialize based on smooth weight's round-to-nearest
        V_init = compute_adaround_v_init(
            module.smooth_weight, module.weight_quantizer,
            zeta, gamma, dev, dtype
        )
        # NOTE: Keep smooth_weight for AdaRound training, will be cleaned up after hardening
    elif adaround_params is not None and name in adaround_params:
        # Load from pre-saved params
        V_init = adaround_params[name].to(device=dev, dtype=dtype)
    else:
        # Fallback to constant init_bias
        V_init = torch.full_like(weight, init_bias, dtype=dtype)

    # Register as parameter
    module.register_parameter('adaround_V', nn.Parameter(V_init))
    module.register_buffer('adaround_zeta', torch.tensor(zeta, device=dev, dtype=dtype))
    module.register_buffer('adaround_gamma', torch.tensor(gamma, device=dev, dtype=dtype))
    module.adaround_enabled = True

    return module.adaround_V



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def affinequant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
    adaround_params=None,
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


        # Check if AdaRound will be applied to this layer
        do_adaround = getattr(args, 'adaround', False) and (getattr(args, 'adaround_layer_idx', None) is None or i == args.adaround_layer_idx)

        # real smooth and quantization
        # Save smooth weights before quantization if AdaRound is enabled
        qlayer.smooth_and_quant_inplace(lm.model.config.num_attention_heads, maskqkv, maskfc,
                                        use_matrix=use_matrix, use_ln_matrix=use_ln_matrix,
                                        save_smooth_weight=do_adaround)

        # === AdaRound Stage ===
        # Only run if --adaround is enabled and (no layer specified or this layer matches)
        if do_adaround:
            logger.info(f"=== Start AdaRound training for layer {i} ===")

            # Freeze let/lwc parameters
            for param in qlayer.let_parameters(use_shift):
                param.requires_grad = False
            for param in qlayer.lwc_parameters():
                param.requires_grad = False

            # IMPORTANT: Enable weight quant so forward uses adaround_fake_quant path
            qlayer.set_quant_state(weight_quant=True, act_quant=True)
            logger.info(f"AdaRound: set weight_quant=True for layer {i}")

            # Initialize adaround params for all QuantLinear in this layer
            # Use smooth weight's round-to-nearest for V initialization
            named_linears = get_named_linears(qlayer)
            adaround_V_list = []
            adaround_module_info = []  # (full_name, module, V)

            for sub_name, module in named_linears.items():
                full_name = f"{layer_name_prefix}.{i}.{sub_name}"
                V = init_adaround_params_for_module(
                    module, full_name, adaround_params,
                    args.adaround_init_bias, args.adaround_zeta, args.adaround_gamma,
                    dev, args.dtype,
                    use_smooth_init=True  # Initialize based on smooth weight's round-to-nearest
                )
                adaround_V_list.append(V)
                adaround_module_info.append((full_name, module, V))
                # Enable adaround mode in quantizer
                module.weight_quantizer.adaround_mode = True

            # Create optimizer for AdaRound V parameters only
            adaround_optimizer = torch.optim.AdamW(adaround_V_list, lr=args.adaround_lr)
            adaround_loss_scaler = utils.NativeScalerWithGradNormCount()

            # Beta annealing: linear decay from beta_start to beta_end
            beta_start = args.adaround_beta_start
            beta_end = args.adaround_beta_end
            adaround_reg_coef = args.adaround_reg

            for adaround_epoch in range(args.adaround_epochs):
                # Compute current beta (annealing)
                if args.adaround_epochs > 1:
                    beta = beta_start + (beta_end - beta_start) * adaround_epoch / (args.adaround_epochs - 1)
                else:
                    beta = beta_end

                adaround_task_loss_list = []
                adaround_reg_loss_list = []
                adaround_total_loss_list = []
                adaround_norm_list = []

                for j in range(args.nsamples // args.batch_size):
                    index = j * args.batch_size

                    with traincast():
                        # Forward with adaround (soft mode)
                        quant_out = qlayer(quant_inps[index:index+args.batch_size,],
                                           attention_mask=attention_mask_batch,
                                           position_ids=position_ids)[0]
                        # Task loss (reconstruction loss)
                        task_loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                        if args.aug_loss:
                            task_loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)

                        # Regularization loss (L_com)
                        reg_loss = 0
                        for _, module, V in adaround_module_info:
                            reg_loss += adaround_reg_loss(
                                V, module.adaround_zeta.item(), module.adaround_gamma.item(), beta
                            )

                        total_loss = task_loss + adaround_reg_coef * reg_loss

                    if not math.isfinite(total_loss.item()):
                        logger.info("AdaRound Loss is NAN, stopping training")
                        pdb.set_trace()

                    adaround_task_loss_list.append(task_loss.data)
                    adaround_reg_loss_list.append(reg_loss.data if isinstance(reg_loss, torch.Tensor) else torch.tensor(reg_loss))
                    adaround_total_loss_list.append(total_loss.data)
                    adaround_optimizer.zero_grad()
                    norm = adaround_loss_scaler(total_loss, adaround_optimizer, parameters=adaround_V_list)
                    adaround_norm_list.append(norm.data)

                task_loss_mean = torch.stack(adaround_task_loss_list).mean()
                reg_loss_mean = torch.stack(adaround_reg_loss_list).mean()
                total_loss_mean = torch.stack(adaround_total_loss_list).mean()
                adaround_norm_mean = torch.stack(adaround_norm_list).mean()
                logger.info(f"layer {i} adaround epoch {adaround_epoch} "
                            f"recon_loss:{task_loss_mean:.10f} reg_loss:{reg_loss_mean:.10f} "
                            f"total_loss:{total_loss_mean:.10f} beta:{beta:.2f} norm:{adaround_norm_mean:.6f}")

            del adaround_optimizer

            # Save adaround parameters before hardening (for analysis/debugging)
            adaround_save_dict = {}
            for full_name, module, V in adaround_module_info:
                adaround_save_dict[full_name] = V.detach().cpu().clone()
            adaround_save_path = os.path.join(args.output_dir, f"adaround_params_layer{i}.pth")
            torch.save(adaround_save_dict, adaround_save_path)
            logger.info(f"Saved adaround params to {adaround_save_path}")

            # Hardening: apply hard rounding and write back to weights
            with torch.no_grad():
                for full_name, module, V in adaround_module_info:
                    # Disable adaround mode in quantizer
                    module.weight_quantizer.adaround_mode = False

                    # Compute hard h
                    h_hard = get_adaround_h(V, module.adaround_zeta.item(),
                                            module.adaround_gamma.item(), hard=True)

                    # Use smooth_weight for final quantization (same as training)
                    smooth_weight = module.smooth_weight if hasattr(module, 'smooth_weight') else module.weight
                    quantizer = module.weight_quantizer

                    # Apply the same reshape/pad logic as fake_quant
                    w = smooth_weight.clone()
                    deficiency = quantizer.deficiency
                    group_size = quantizer.group_size

                    if deficiency > 0:
                        pad_zeros = torch.zeros((w.shape[0], deficiency), dtype=w.dtype, device=w.device)
                        w = torch.cat((w, pad_zeros), dim=1)

                    # Prepare h for reshape
                    if deficiency > 0:
                        h_pad = torch.zeros((h_hard.shape[0], deficiency), dtype=h_hard.dtype, device=h_hard.device)
                        h_padded = torch.cat((h_hard, h_pad), dim=1)
                    else:
                        h_padded = h_hard

                    if group_size:
                        dim1, dim2 = w.shape
                        w = w.reshape(-1, group_size)
                        h_flat = h_padded.reshape(-1, group_size)
                    else:
                        h_flat = h_padded

                    # Calibrate scale/zero using smooth weight
                    quantizer.per_token_dynamic_calibration(w)
                    scale = quantizer.scale
                    round_zero_point = quantizer.round_zero_point

                    # floor(w/scale) + h_hard
                    w_floor = torch.floor(w / scale)
                    w_int = w_floor + h_flat
                    if round_zero_point is not None:
                        w_int = w_int.add(round_zero_point)
                    w_int = w_int.clamp(quantizer.qmin, quantizer.qmax)

                    # Dequant
                    w_dequant = w_int
                    if round_zero_point is not None:
                        w_dequant = w_dequant.sub(round_zero_point)
                    w_dequant = w_dequant.mul(scale)

                    if group_size:
                        w_dequant = w_dequant.reshape(dim1, dim2)
                    if deficiency > 0:
                        w_dequant = w_dequant[:, :-deficiency]

                    # Write back to weight buffer
                    module.weight.copy_(w_dequant)

                    # IMPORTANT: Disable weight quantization since weight is already quantized
                    # This prevents the weight_quantizer from re-quantizing the AdaRound-optimized weight
                    module.use_weight_quant = False

                    # Clean up adaround params and smooth_weight
                    del module.adaround_V
                    del module.adaround_zeta
                    del module.adaround_gamma
                    if hasattr(module, 'smooth_weight'):
                        del module.smooth_weight
                    module.adaround_enabled = False

            # Restore quant state: disable weight quant since weights are already quantized
            qlayer.set_quant_state(weight_quant=False, act_quant=True)
            logger.info(f"=== AdaRound hardening done for layer {i}, set weight_quant=False ===")

            # Re-enable let/lwc parameters (in case needed later)
            for param in qlayer.let_parameters(use_shift):
                param.requires_grad = True
            for param in qlayer.lwc_parameters():
                param.requires_grad = True

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

