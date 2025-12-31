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


def adaround_reg_loss(A1, A2, zeta, gamma, beta):
    """
    Compute LoRA-AdaRound regularization loss L_com.
    V = A1 @ A2
    L_com = sum(1 - |2 * h(V) - 1|^beta)
    where h(V) = clamp(sigmoid(V) * (zeta - gamma) + gamma, 0, 1)
    """
    V = A1 @ A2
    h = torch.clamp(torch.sigmoid(V) * (zeta - gamma) + gamma, 0, 1)
    reg = 1 - torch.pow((2 * h - 1).abs(), beta)
    return reg.sum()


def get_adaround_h(A1, A2, zeta, gamma, hard=False):
    """
    Compute LoRA-AdaRound soft/hard rounding offset h(V).
    V = A1 @ A2
    h = clamp(sigmoid(V) * (zeta - gamma) + gamma, 0, 1)
    If hard=True, return (h >= 0.5).float()
    """
    V = A1 @ A2
    h = torch.clamp(torch.sigmoid(V) * (zeta - gamma) + gamma, 0, 1)
    if hard:
        h = (h >= 0.5).float()
    return h


def init_adaround_params_for_module(module, name, adaround_params, init_bias, zeta, gamma, rank, dev, dtype):
    """
    Initialize LoRA-AdaRound parameters for a QuantLinear module.
    V = A1 @ A2 where A1: (out_features, rank), A2: (rank, in_features)
    A1: Gaussian random init, A2: zero init (so initial V = 0, h = 0.5 = round-to-nearest)
    Returns tuple (A1, A2) parameter tensors.
    """
    if hasattr(module, 'adaround_enabled') and module.adaround_enabled:
        return module.adaround_A1, module.adaround_A2

    out_features, in_features = module.weight.shape

    # LoRA-style decomposition: V = A1 @ A2
    # A1: (out_features, rank) - Gaussian random init
    # A2: (rank, in_features) - zero init
    # Initial V = A1 @ A2 = 0, so h = sigmoid(0) * (zeta - gamma) + gamma = 0.5
    if adaround_params is not None and f"{name}.A1" in adaround_params:
        A1_init = adaround_params[f"{name}.A1"].to(device=dev, dtype=dtype)
        A2_init = adaround_params[f"{name}.A2"].to(device=dev, dtype=dtype)
        # Infer rank from loaded A1 matrix shape
        actual_rank = A1_init.shape[1]
    else:
        A1_init = torch.randn(out_features, rank, device=dev, dtype=dtype) * 0.01
        A2_init = torch.zeros(rank, in_features, device=dev, dtype=dtype)
        actual_rank = rank

    # Register as parameters
    module.register_parameter('adaround_A1', nn.Parameter(A1_init))
    module.register_parameter('adaround_A2', nn.Parameter(A2_init))
    module.register_buffer('adaround_zeta', torch.tensor(zeta, device=dev, dtype=dtype))
    module.register_buffer('adaround_gamma', torch.tensor(gamma, device=dev, dtype=dtype))
    module.register_buffer('adaround_rank', torch.tensor(actual_rank, device=dev))
    module.adaround_enabled = True

    return module.adaround_A1, module.adaround_A2



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

        # Check if AdaRound will be applied to this layer (for joint optimization)
        do_adaround = getattr(args, 'adaround', False) and (getattr(args, 'adaround_layer_idx', None) is None or i == args.adaround_layer_idx)

        # === Initialize AdaRound params BEFORE training (for joint optimization) ===
        adaround_params_list = []  # List of (A1, A2) tuples for optimizer
        adaround_module_info = []  # (full_name, module, A1, A2)
        if do_adaround and args.epochs > 0 and not (args.resume and i < len(affine_parameters)):
            logger.info(f"=== Initializing LoRA-AdaRound (rank={args.adaround_rank}) for joint optimization in layer {i} ===")
            named_linears = get_named_linears(qlayer)
            for sub_name, module in named_linears.items():
                full_name = f"{layer_name_prefix}.{i}.{sub_name}"
                A1, A2 = init_adaround_params_for_module(
                    module, full_name, adaround_params,
                    args.adaround_init_bias, args.adaround_zeta, args.adaround_gamma,
                    args.adaround_rank, dev, args.dtype
                )
                adaround_params_list.extend([A1, A2])
                adaround_module_info.append((full_name, module, A1, A2))
                # Enable adaround mode in quantizer for forward pass
                module.weight_quantizer.adaround_mode = True

        if args.epochs > 0 and not (args.resume and i < len(affine_parameters)):
            with torch.no_grad():
                qlayer.to(args.dtype)      # required for AMP training

            # Create separate optimizers for AffineQuant and AdaRound (alternating training)
            affine_optimizer = torch.optim.AdamW([
                {"params": qlayer.let_parameters(use_shift), "lr": args.let_lr},
                {"params": qlayer.lwc_parameters(), "lr": args.lwc_lr}
            ], weight_decay=args.wd)

            if do_adaround and len(adaround_params_list) > 0:
                adaround_optimizer = torch.optim.AdamW([
                    {"params": adaround_params_list, "lr": args.adaround_lr}
                ], weight_decay=args.wd)
                adaround_inner_epochs = getattr(args, 'adaround_inner_epochs', 3)
            else:
                adaround_optimizer = None
                adaround_inner_epochs = 0

            loss_scaler = utils.NativeScalerWithGradNormCount()

            # Beta annealing parameters for AdaRound regularization
            beta_start = args.adaround_beta_start if do_adaround else 20.0
            beta_end = args.adaround_beta_end if do_adaround else 2.0
            adaround_reg_coef = args.adaround_reg if do_adaround else 0.01

            # Total AdaRound epochs for beta annealing calculation
            total_adaround_epochs = args.epochs * adaround_inner_epochs if do_adaround else 1
            adaround_epoch_counter = 0

            for epochs in range(args.epochs):
                # gradual mask (same for both phases)
                qkvmask_num = int((lm.model.config.hidden_size-1)/(args.epochs-1)*epochs)+1 if args.epochs > 1 else lm.model.config.hidden_size
                fc1mask_num = int((lm.model.config.hidden_size/lm.model.config.num_attention_heads-1)/(args.epochs-1)*epochs)+1 if args.epochs > 1 else lm.model.config.hidden_size//lm.model.config.num_attention_heads

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

                # ============ Phase 1: Train AffineQuant (LET/LWC) for 1 epoch ============
                loss_list = []
                norm_list = []

                for j in range(args.nsamples//args.batch_size):
                    index = j * args.batch_size
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
                    affine_optimizer.zero_grad()
                    norm = loss_scaler(loss, affine_optimizer, parameters=list(qlayer.affine_parameters(use_shift)))
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} epoch {epochs} [AffineQuant] loss:{loss_mean:.6f} norm:{norm_mean:.8f} mem:{torch.cuda.max_memory_allocated(lm._device) / 1024**2:.1f}MB")

                # ============ Phase 2: Train LoRA-AdaRound for inner_epochs ============
                if do_adaround and adaround_optimizer is not None:
                    for inner_epoch in range(adaround_inner_epochs):
                        loss_list = []
                        norm_list = []
                        reg_loss_list = []

                        # Compute current beta for AdaRound regularization (annealing across all adaround epochs)
                        if total_adaround_epochs > 1:
                            beta = beta_start + (beta_end - beta_start) * adaround_epoch_counter / (total_adaround_epochs - 1)
                        else:
                            beta = beta_end

                        for j in range(args.nsamples//args.batch_size):
                            index = j * args.batch_size
                            with traincast():
                                qlayer.smooth_and_quant_temporary(lm.model.config.num_attention_heads, maskqkv, maskfc, use_matrix=use_matrix, use_ln_matrix=use_ln_matrix)
                                quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                                recon_loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                                if args.aug_loss:
                                    recon_loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)

                                # AdaRound regularization loss L_com
                                reg_loss = 0
                                for _, module, A1, A2 in adaround_module_info:
                                    reg_loss += adaround_reg_loss(
                                        A1, A2, module.adaround_zeta.item(), module.adaround_gamma.item(), beta
                                    )
                                loss = recon_loss + adaround_reg_coef * reg_loss
                                reg_loss_list.append(reg_loss.data if isinstance(reg_loss, torch.Tensor) else torch.tensor(reg_loss))

                            if not math.isfinite(loss.item()):
                                logger.info("Loss is NAN, stopping training")
                                pdb.set_trace()

                            loss_list.append(loss.data)
                            adaround_optimizer.zero_grad()
                            norm = loss_scaler(loss, adaround_optimizer, parameters=adaround_params_list)
                            norm_list.append(norm.data)

                        loss_mean = torch.stack(loss_list).mean()
                        norm_mean = torch.stack(norm_list).mean()
                        reg_mean = torch.stack(reg_loss_list).mean()
                        logger.info(f"layer {i} epoch {epochs} [AdaRound {inner_epoch+1}/{adaround_inner_epochs}] loss:{loss_mean:.6f} reg:{reg_mean:.6f} beta:{beta:.2f} norm:{norm_mean:.4f}")

                        adaround_epoch_counter += 1

            qlayer.clear_temp_variable()
            del affine_optimizer
            if adaround_optimizer is not None:
                del adaround_optimizer

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
        # When adaround is enabled, skip RTN quantization - hardening will do the final quant
        skip_rtn_quant = do_adaround and len(adaround_module_info) > 0
        qlayer.smooth_and_quant_inplace(lm.model.config.num_attention_heads, maskqkv, maskfc,
                                        use_matrix=use_matrix, use_ln_matrix=use_ln_matrix,
                                        save_smooth_weight=False, skip_quant=skip_rtn_quant)

        # === AdaRound Hardening (after joint training) ===
        # Only run if --adaround is enabled and training was performed
        if do_adaround and len(adaround_module_info) > 0:
            logger.info(f"=== LoRA-AdaRound hardening for layer {i} (after joint training) ===")

            # Save adaround parameters before hardening (A1 and A2 separately)
            adaround_save_dict = {}
            for full_name, module, A1, A2 in adaround_module_info:
                adaround_save_dict[f"{full_name}.A1"] = A1.detach().cpu().clone()
                adaround_save_dict[f"{full_name}.A2"] = A2.detach().cpu().clone()
            adaround_save_path = os.path.join(args.output_dir, f"adaround_params_layer{i}.pth")
            torch.save(adaround_save_dict, adaround_save_path)
            logger.info(f"Saved LoRA-AdaRound params to {adaround_save_path}")

            # Hardening: apply hard rounding and write back to weights
            with torch.no_grad():
                for full_name, module, A1, A2 in adaround_module_info:
                    # Disable adaround mode in quantizer
                    module.weight_quantizer.adaround_mode = False

                    # Compute hard h from V = A1 @ A2
                    h_hard = get_adaround_h(A1, A2, module.adaround_zeta.item(),
                                            module.adaround_gamma.item(), hard=True)

                    # Use current weight (already smoothed) for final quantization
                    quantizer = module.weight_quantizer
                    w = module.weight.clone()
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

                    # Calibrate scale/zero using weight
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
                    module.use_weight_quant = False

                    # Clean up adaround params (A1, A2 instead of V)
                    del module.adaround_A1
                    del module.adaround_A2
                    del module.adaround_zeta
                    del module.adaround_gamma
                    del module.adaround_rank
                    module.adaround_enabled = False

            # Set quant state: disable weight quant since weights are already quantized
            qlayer.set_quant_state(weight_quant=False, act_quant=True)
            logger.info(f"=== LoRA-AdaRound hardening done for layer {i} ===")

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

    # Consolidate and save all adaround params if adaround was used
    if getattr(args, 'adaround', False):
        consolidated_adaround = {}
        for layer_idx in range(len(layers)):
            layer_file = os.path.join(args.output_dir, f"adaround_params_layer{layer_idx}.pth")
            if os.path.exists(layer_file):
                layer_params = torch.load(layer_file, weights_only=False)
                consolidated_adaround.update(layer_params)
        if consolidated_adaround:
            save_path = os.path.join(args.output_dir, "adaround_params.pth")
            torch.save(consolidated_adaround, save_path)
            logger.info(f"Saved consolidated adaround params to {save_path}")

    model.config.use_cache = use_cache
    return model.half()

