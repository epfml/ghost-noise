import torch
import torch.nn as nn
from torch.distributions.chi2 import Chi2


class NoisyAffineTransform(nn.Module):
    def __init__(
        self,
        batch_size,
        scale_noise=True,
        shift_noise=True,
        scale_shift=True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.scale_noise = scale_noise
        self.shift_noise = shift_noise
        self.scale_shift = scale_shift  # If standard deviation is not already one

    def forward(self, x):
        if self.training:
            N = self.batch_size
            noise_shape = (*x.shape[:2], *[1]*(x.dim()-2))

            if self.shift_noise:
                shift_noise = torch.randn(noise_shape, device=x.device) / N**0.5
                if self.scale_shift:
                    # Unfortunately needed unless we perform where already normalized
                    shift_noise = shift_noise * x.std(dim=[2, 3], keepdim=True, correction=0)
                x = x - shift_noise

            if self.scale_noise:
                scale_noise = (Chi2(torch.full(noise_shape,N-1,device=x.device)).sample() / N)**0.5
                x = x / scale_noise

        return x

    def extra_repr(self):
        s = "batch_size={batch_size}"
        s += ", scale_noise={scale_noise}"
        s += ", shift_noise={shift_noise}"
        s += ", scale_shift={scale_shift}"
        return s.format(**self.__dict__)
