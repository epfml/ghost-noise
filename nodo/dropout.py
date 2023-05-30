import torch
import torch.nn as nn

class FlexibleDropout(nn.Module):
    def __init__(
        self,
        drop_probability,
        noise_type='bernoulli',
        channel_wise=False,
        scaling_type='mean',
        max_steps=float('inf'),
    ):
        super().__init__()
        self.p = 1 - drop_probability  # keep probability
        self.noise_type = noise_type.lower()  # bernoulli or gaussian
        assert noise_type in ['bernoulli', 'gaussian']
        self.channel_wise = channel_wise
        self.scaling_type = scaling_type.lower()  # mean or none
        assert self.scaling_type in ['mean', 'none']
        self.max_steps = max_steps  # turn off after x steps
        self.current_iteration = 0  # only counts training steps

    def forward(self, x):
        if self.training and self.current_iteration < self.max_steps and self.p < 1.0:
            self.current_iteration += 1

            if self.channel_wise:
                noise_shape = (x.shape[0], x.shape[1], *([1]*(x.dim()-2)))
            else:
                noise_shape = x.shape

            if self.noise_type == 'bernoulli':
                noise = torch.bernoulli(torch.full(noise_shape, self.p, device=x.device))
                if self.scaling_type == 'mean':
                    noise = noise / self.p
            elif self.noise_type == 'gaussian':
                noise = torch.normal(1.0, (1-self.p)/self.p, noise_shape, device=x.device)
                if self.scaling_type == 'mean':
                    # No scaling for mean required, i.e. implicitly scaled by 1
                    pass

            x = x * noise
        return x

    def extra_repr(self):
        s = "keep_probability={p}, noise_type={noise_type}"
        s += ", channel_wise={channel_wise}, scaling_type={scaling_type}"
        s += ", current_iteration={current_iteration}" if self.max_steps < float('inf') else ""
        s += ", max_steps={max_steps}" if self.max_steps < float('inf') else ""
        return s.format(**self.__dict__)
