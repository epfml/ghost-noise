import torch
import torch.nn as nn

class ExclusiveBatchNorm2d(nn.Module):
    def __init__(
        self,
        num_features,
        batch_size=None,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        device=None,
        dtype=None,
        checkpoint=False,
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.batch_size = batch_size
        self.checkpoint = checkpoint

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
            with torch.no_grad():
                var, mean = torch.var_mean(
                    x.flatten(2).to(dtype=self.running_mean.dtype),
                    dim=(0,2), correction=0, keepdim=False,
                )
                self.running_mean.copy_(torch.lerp(self.running_mean, mean, self.momentum))
                self.running_var.copy_(torch.lerp(self.running_var, var, self.momentum))

            if self.checkpoint:
                # Reduce memory overhead due to the lack of kernel / autograd function
                x = torch.utils.checkpoint.checkpoint(xbn2d, x, self.weight, self.bias, self.batch_size, self.eps)
            else:
                x = xbn2d(x, self.weight, self.bias, self.batch_size, self.eps)
        else:
            # # Use BN kernel here for efficiency
            x = torch.nn.functional.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                weight=self.weight,
                bias=self.bias,
                training=False,
                momentum=0.1,
                eps=1e-05,
            )

        return x

    def extra_repr(self):
        s = "{num_features}, "
        s += "batch_size={batch_size}, " if self.batch_size else ""
        s += "eps={eps}, momentum={momentum}, affine={affine}"
        return s.format(**self.__dict__)


def xbn2d(x, weight, bias, batch_size=None, eps=1e-5):
    N = batch_size or x.shape[0]
    xx = x.view(-1, N, *x.shape[1:])

    sample_means = xx.mean(dim=(-2, -1), keepdim=True, dtype=torch.float32) # Mean over H, W
    mu = (sample_means.sum(dim=1, keepdim=True) - sample_means) / (N-1)

    # The multiplication here is a bit sketchy in lower precision but saves some memory
    sample_square_means = (xx*xx).mean(dim=(-2, -1), keepdim=True, dtype=torch.float32)
    sigma2 = (sample_square_means.sum(dim=1, keepdim=True) - sample_square_means) / (N-1) - mu * mu

    denom = torch.sqrt(sigma2 + eps).view(*x.shape[:2], 1, 1)

    if weight is not None:
        # Try to avoid extra memory copies including HxW to save memory
        gain = weight.view(1, -1, 1, 1) / denom
        shift = bias.view(1, -1, 1, 1) - mu.view(*x.shape[:2], 1, 1) * gain
        x = torch.addcmul(
            shift,
            gain,
            x,
        ).to(dtype=x.dtype)
    else:
        xx = (xx - mu) / torch.sqrt(sigma2 + eps)
        x = xx.view_as(x).to(dtype=x.dtype)

    return x
