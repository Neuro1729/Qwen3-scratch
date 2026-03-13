import torch
import math

def rmsnorm(x, weight, eps=1e-6):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight


def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, seq_len, base=10000):
    # q, k: [B, H, T, D] or [T, H, D] (support both)
    has_batch = q.dim() == 4  # True if [B, H, T, D]
    
    if not has_batch:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)

    B, H, T, D = q.shape
    device = q.device

    pos = torch.arange(seq_len, device=device)              # [T]
    dim = torch.arange(0, D, 2, device=device)             # [D/2]
    inv_freq = 1.0 / (base ** (dim / D))                   # [D/2]
    angles = pos[:, None] * inv_freq[None, :]              # [T, D/2]

    sin = torch.sin(angles)                                 # [T, D/2]
    cos = torch.cos(angles)                                 # [T, D/2]

    # Expand to full dimension
    sin = torch.repeat_interleave(sin, 2, dim=-1)          # [T, D]
    cos = torch.repeat_interleave(cos, 2, dim=-1)          # [T, D]

    # Expand for heads and batch
    sin = sin[None, None, :, :].expand(B, H, T, D)         # [B, H, T, D]
    cos = cos[None, None, :, :].expand(B, H, T, D)         # [B, H, T, D]

    # Apply RoPE
    q = q * cos + rotate_half(q) * sin                     # [B, H, T, D]
    k = k * cos + rotate_half(k) * sin                     # [B, H, T, D]

    if not has_batch:
        q = q.squeeze(0)
        k = k.squeeze(0)

    return q, k
