import torch
import torch.nn as nn

class GhostNoiseInjectorR(nn.Module):
    def __init__(
        self,
        batch_size,
        eps=1e-3,
        scale_noise=True,
        shift_noise=True,
        channel_last=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.eps = eps
        self.scale_noise = scale_noise
        self.shift_noise = shift_noise
        self.channel_last = channel_last

    def forward(self, x):
        if self.training:
            N = self.batch_size

            if self.channel_last:
                x = x.transpose(1, -1)

            # Sample with replacement, allow any batch size (always "shuffle")
            # Here we randomly select samples, not channels within a sample (alternative)

            with torch.no_grad():
                idxs = torch.randint(x.shape[0], (x.shape[0], N))

                instance_mean = torch.mean(x.float().reshape(*x.shape[:2], -1), dim=2)
                instance_mean2 = torch.mean(x.float().reshape(*x.shape[:2], -1)**2, dim=2)

                full_mean = torch.mean(instance_mean, dim=0)
                full_var = torch.mean(instance_mean2, dim=0) - full_mean**2

                ghost_mean = instance_mean[idxs].mean(dim=1)
                ghost_mean2 = instance_mean2[idxs].mean(dim=1)
                ghost_var = ghost_mean2 - ghost_mean**2

                shift_noise = (ghost_mean - full_mean.reshape(1, -1)).to(dtype=x.dtype)
                scale_noise = torch.sqrt(
                    (ghost_var + self.eps)/(full_var.view(1, -1) + self.eps)
                ).to(dtype=x.dtype)

            if self.shift_noise:
                x = x - shift_noise.reshape(*x.shape[:2], *[1]*(x.dim()-2))
            if self.scale_noise:
                x = x / scale_noise.reshape(*x.shape[:2], *[1]*(x.dim()-2))

            if self.channel_last:
                x = x.transpose(1, -1)

        return x

    def extra_repr(self):
        s = "batch_size={batch_size}, eps={eps}"
        s += ", scale_noise={scale_noise}" if not self.scale_noise else ""
        s += ", shift_noise={shift_noise}" if not self.shift_noise else ""
        s += ", channel_last={channel_last}"
        return s.format(**self.__dict__)
