# Minimal implementation of Ghost Noise Injection
# For our experiments we use the implmentation in 
# ghost_noise_injector_replacement.py

import torch

def ghost_noise_injection(X: torch.Tensor, N: int, eps: float=1e-3):
    B, C, _, _ = X.shape  # Shape: Batch (B), Channels (C), Height (H), Width (W)
    with torch.no_grad():
        batch_var, batch_mean = torch.var_mean(X, dim=(0,2,3), correction=0)
        ghost_var = torch.zeros(size=(B,C), device=X.device)
        ghost_mean = torch.zeros(size=(B,C), device=X.device)
        for idx in range(B):
            ghost_sample = torch.randint(high=B, size=(N,))
            ghost_var[idx], ghost_mean[idx] = torch.var_mean(
                X[ghost_sample], dim=[0,2,3], correction=0)
        shift = (ghost_mean - batch_mean)
        scale = torch.sqrt((ghost_var + eps)/(batch_var + eps))
    return (X - shift.view(B, C, 1, 1)) / scale.view(B, C, 1, 1)
