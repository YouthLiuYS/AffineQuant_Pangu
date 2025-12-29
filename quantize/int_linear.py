import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer


def adaround_fake_quant(x, quantizer, V, zeta, gamma):
    """
    Perform AdaRound quantization with soft rounding.
    x: input weight tensor
    quantizer: UniformAffineQuantizer instance
    V: learnable rounding parameter (same shape as x)
    zeta, gamma: rectified sigmoid parameters
    """
    deficiency = quantizer.deficiency
    group_size = quantizer.group_size

    # Pad if needed
    if deficiency > 0:
        pad_zeros = torch.zeros((x.shape[0], deficiency), dtype=x.dtype, device=x.device)
        x_padded = torch.cat((x, pad_zeros), dim=1)
        V_pad = torch.zeros((V.shape[0], deficiency), dtype=V.dtype, device=V.device)
        V_padded = torch.cat((V, V_pad), dim=1)
    else:
        x_padded = x
        V_padded = V

    # Reshape for group quantization
    if group_size:
        dim1, dim2 = x_padded.shape
        x_reshaped = x_padded.reshape(-1, group_size)
        V_reshaped = V_padded.reshape(-1, group_size)
    else:
        x_reshaped = x_padded
        V_reshaped = V_padded

    # Calibrate scale and zero point
    quantizer.per_token_dynamic_calibration(x_reshaped)
    scale = quantizer.scale
    round_zero_point = quantizer.round_zero_point

    # AdaRound soft rounding: floor(x/scale) + h(V)
    # h(V) = clamp(sigmoid(V) * (zeta - gamma) + gamma, 0, 1)
    x_floor = torch.floor(x_reshaped / scale)
    h = torch.clamp(torch.sigmoid(V_reshaped) * (zeta - gamma) + gamma, 0, 1)
    x_int = x_floor + h

    if round_zero_point is not None:
        x_int = x_int.add(round_zero_point)
    x_int = x_int.clamp(quantizer.qmin, quantizer.qmax)

    # Dequantize
    x_dequant = x_int
    if round_zero_point is not None:
        x_dequant = x_dequant.sub(round_zero_point)
    x_dequant = x_dequant.mul(scale)

    # Reshape back
    if group_size:
        x_dequant = x_dequant.reshape(dim1, dim2)
    if deficiency > 0:
        x_dequant = x_dequant[:, :-deficiency]

    return x_dequant






class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_buffer('weight',org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False

    
    
    def forward(self, input: torch.Tensor):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            # Check for AdaRound mode
            if (self.weight_quantizer.adaround_mode and
                hasattr(self, 'adaround_enabled') and self.adaround_enabled):
                weight = adaround_fake_quant(
                    self.weight,
                    self.weight_quantizer,
                    self.adaround_V,
                    self.adaround_zeta.item(),
                    self.adaround_gamma.item()
                )
            else:
                weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
