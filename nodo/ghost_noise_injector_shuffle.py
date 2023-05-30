# This is an older version of GNI that shuffles the data instead of sampling with replacement

import torch
import torch.nn as nn

class GhostNoiseInjectorS(nn.Module):
    def __init__(
        self,
        batch_size,
        eps=1e-6,  # 1e-3
        shuffle=False,
        scale_noise=True,
        shift_noise=True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.eps = eps
        self.shuffle = shuffle
        self.scale_noise = scale_noise
        self.shift_noise = shift_noise

    def forward(self, x):
        if self.training:
            N = self.batch_size

            with torch.no_grad():
                if self.shuffle:
                    idx = torch.randperm(x.size(0))
                    xx = x[idx]
                    xx = xx.view(-1, N, *x.shape[1:])
                else:
                    xx = x.view(-1, N, *x.shape[1:])

                # Fold x into G, N, C, H, W where N is the normalization batch size
                full_var, full_mean = torch.var_mean(
                    xx,
                    dim=(0,1,3,4),
                    keepdim=True,
                    correction=0,
                )

                ghost_var, ghost_mean = torch.var_mean(
                    xx,
                    dim=(1,3,4),
                    keepdim=True,
                    correction=0,
                )

                shift_noise = ghost_mean - full_mean
                # Slightly different form for numerical stability (placement of eps)
                scale_noise = torch.sqrt((ghost_var + self.eps)/(full_var + self.eps))

            if self.shift_noise:
                x = (x.view_as(xx) - shift_noise).view_as(x)
            if self.scale_noise:
                x = (x.view_as(xx) / scale_noise).view_as(x)

        return x

    def extra_repr(self):
        s = "batch_size={batch_size}, eps={eps}, shuffle={shuffle}"
        s += ", scale_noise={scale_noise}" if not self.scale_noise else ""
        s += ", shift_noise={shift_noise}" if not self.shift_noise else ""
        return s.format(**self.__dict__)
