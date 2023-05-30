import torch
import torch.nn as nn

class EvalNorm2d(nn.Module):
    def __init__(
        self,
        num_features,
        batch_size=None,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.batch_size = batch_size

        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_features, device=device, dtype=dtype))
            self.bias = nn.Parameter(torch.empty(num_features, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.register_buffer('running_mean', torch.zeros(num_features, device=device, dtype=dtype))
        self.register_buffer('running_var', torch.ones(num_features, device=device, dtype=dtype))

        self.reset_parameters()

    def reset_running_stats(self) -> None:
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.training:
            # Fold x into G, N, C, H, W where N is the normalization batch size
            N = self.batch_size or x.shape[0]
            x = x.view(-1, N, *x.shape[1:])
            mu = x.mean(dim=(1,3,4), keepdim=True)
            sigma2 = (x*x).mean(dim=(1,3,4), keepdim=True) - mu * mu

            x = (x - mu) / torch.sqrt(sigma2 + self.eps)
            x = x.view(-1, *x.shape[2:])

            with torch.no_grad():
                new_mu = self.running_mean * (1-self.momentum) + self.momentum * mu.mean(dim=0).flatten()
                self.running_mean.copy_(new_mu)
                new_var = self.running_var * (1-self.momentum) + self.momentum * sigma2.mean(dim=0).flatten()
                self.running_var.copy_(new_var)
        else:
            self_sigma2, self_mu = torch.var_mean(x, dim=(2,3), correction=0, keepdim=True)

            ab = 1/self.batch_size**2
            mu = torch.lerp(self.running_mean.view(1, -1, 1, 1), self_mu, ab)
            sigma2 = torch.lerp(self.running_var.view(1, -1, 1, 1), self_sigma2, ab)
            sigma2 += ab*(1-ab) * (self_mu - self.running_mean.view(1, -1, 1, 1))**2

            x = (x - mu)/torch.sqrt(sigma2 + self.eps)

        if self.weight is not None:
            x *= self.weight.view(1, -1, 1, 1)
        if self.bias is not None:
            x += self.bias.view(1, -1, 1, 1)

        return x

    def extra_repr(self):
        s = "{num_features}"
        s += ", batch_size={batch_size}" if self.batch_size else ""
        s += ", eps={eps}, momentum={momentum}, affine={affine}"
        return s.format(**self.__dict__)
