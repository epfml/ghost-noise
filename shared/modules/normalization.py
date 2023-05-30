import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class WSConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 device=None, dtype=None, gain=True, keep_init=True):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        if gain:
            self.gain = nn.Parameter(torch.ones(out_channels, device=device, dtype=dtype))
        else:
            self.register_parameter("gain", None)

        # Keep the initialization magnitude, otherwise use fan-in
        self.keep_init = keep_init
        self.buffer_initialized = False
        if self.keep_init:
            self.register_buffer('init_std', torch.zeros(out_channels, device=device, dtype=dtype))

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=[1, 2, 3], keepdim=True)
        weight = weight - weight_mean
        std = weight.std(dim=[1, 2, 3], keepdim=True) + 1e-5

        if self.keep_init and not self.buffer_initialized:
            with torch.no_grad():
                self.init_std.copy_(std.flatten())
            self.buffer_initialized = True
            scale_factor = self.init_std.view(-1, 1, 1, 1) / std
        else:
            fan_in = weight.size(1) * weight.size(2) * weight.size(3)
            scale_factor = 1.0 / (std * math.sqrt(fan_in))

        if self.gain is not None:
            scale_factor = scale_factor * self.gain.view(-1, 1, 1, 1)
        weight = scale_factor * weight  # Could also apply to outputs, note different memory impact
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class WSLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, gain=True, keep_init=True):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        if gain:
            self.gain = nn.Parameter(torch.ones(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("gain", None)

        # Keep the initialization magnitude, otherwise use fan-in
        self.keep_init = keep_init
        self.buffer_initialized = False
        if self.keep_init:
            self.register_buffer('init_std', torch.zeros(out_features, device=device, dtype=dtype))

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=[1], keepdim=True)
        weight = weight - weight_mean
        std = weight.std(dim=[1], keepdim=True) + 1e-5

        if self.keep_init and not self.buffer_initialized:
            with torch.no_grad():
                self.init_std.copy_(std.flatten())
                self.buffer_initialized = True
            self.buffer_initialized = True
            scale_factor = self.init_std.view(-1, 1) / std
        else:
            fan_in = weight.shape[1]
            scale_factor = 1.0 / (std * math.sqrt(fan_in))

        if self.gain is not None:
            scale_factor = scale_factor * self.gain.view(-1, 1)
        weight = scale_factor * weight  # Could also apply to outputs, note different memory impact
        return F.linear(x, weight, self.bias)


class Affine(torch.nn.Module):
    def __init__(self, num_features, weight=True, bias=True, device=None, dtype=None):
        super().__init__()

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(num_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        if weight:
            self.weight = torch.nn.Parameter(torch.ones(num_features, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        if self.weight is not None:
            x *= self.weight.view(1, -1, 1, 1)
        if self.bias is not None:
            x += self.bias.view(1, -1, 1, 1)
        return x

    def extra_repr(self):
        s = "{num_features}"
        return s.format(**self.__dict__)


class LayerNorm(torch.nn.LayerNorm):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, num_features, bias):
        super().__init__(num_features)
        self.weight = torch.nn.Parameter(torch.ones(num_features))
        self.bias = torch.nn.Parameter(torch.zeros(num_features)) if bias else None

    def forward(self, input):
        return torch.nn.functional.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
