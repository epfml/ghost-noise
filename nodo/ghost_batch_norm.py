import torch
import torch.nn as nn

class GhostBatchNorm2d(nn.Module):
    def __init__(
        self,
        num_features,
        batch_size=None,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        device=None,
        dtype=None,
        stop_mu_sigma_grad=False,
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.batch_size = batch_size
        self.stop_mu_sigma_grad = stop_mu_sigma_grad

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
            if self.stop_mu_sigma_grad:
                mu = mu.detach()
                sigma2 = sigma2.detach()

            x = (x - mu) / torch.sqrt(sigma2 + self.eps)
            x = x.view(-1, *x.shape[2:])

            with torch.no_grad():
                new_mu = self.running_mean * (1-self.momentum) + self.momentum * mu.mean(dim=0).flatten()
                self.running_mean.copy_(new_mu)
                new_var = self.running_var * (1-self.momentum) + self.momentum * sigma2.mean(dim=0).flatten()
                self.running_var.copy_(new_var)
        else:
            x = x - self.running_mean.view(1, -1, 1, 1)
            x = x / torch.sqrt(self.running_var.view(1, -1, 1, 1) + self.eps)

        # Scale and Bias
        if self.weight is not None:
            x *= self.weight.view(1, -1, 1, 1)
        if self.bias is not None:
            x += self.bias.view(1, -1, 1, 1)

        return x

    def extra_repr(self):
        s = "{num_features}"
        s += ", batch_size={batch_size}" if self.batch_size else ""
        s += ", eps={eps}, momentum={momentum}, affine={affine}"
        s += ", stop_mu_sigma_grad={stop_mu_sigma_grad}" if self.stop_mu_sigma_grad else ""
        return s.format(**self.__dict__)
