"""Image-only rank-eight residuals on raw native GDN heads zero and one."""
import torch
from torch import nn


class GeoAddress(nn.Module):
    def __init__(self, hidden_size=2560, head_dim=128, rank=8, seed=429):
        super().__init__()
        self.hidden_size, self.head_dim = hidden_size, head_dim
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.q_down = nn.Linear(hidden_size, rank, bias=False)
            self.q_up = nn.Linear(rank, 2 * head_dim, bias=False)
            self.k_down = nn.Linear(hidden_size, rank, bias=False)
            self.k_up = nn.Linear(rank, 2 * head_dim, bias=False)
            nn.init.normal_(self.q_down.weight, std=.02)
            nn.init.normal_(self.k_down.weight, std=.02)
            nn.init.zeros_(self.q_up.weight)
            nn.init.zeros_(self.k_up.weight)

    def forward(self, x, q_raw, k_raw, image_mask):
        if (x.ndim != 3 or x.shape[-1] != self.hidden_size or q_raw.ndim != 4
                or q_raw.shape != k_raw.shape or q_raw.shape[:2] != x.shape[:2]
                or q_raw.shape[2] < 2 or q_raw.shape[3] != self.head_dim):
            raise ValueError('GeoAddress hidden/raw head dimensions disagree')
        if (image_mask is None or image_mask.dtype != torch.bool
                or image_mask.shape != x.shape[:2] or image_mask.device != x.device):
            raise ValueError('GeoAddress requires explicit boolean [B,L] image mask on device')
        result = []
        for raw, down, up in ((q_raw, self.q_down, self.q_up), (k_raw, self.k_down, self.k_up)):
            delta = up(down(x)).reshape(*x.shape[:2], 2, self.head_dim)
            if not torch.isfinite(delta).all():
                raise FloatingPointError('Nonfinite GeoAddress residual')
            selected = raw[:, :, :2] + delta * image_mask[:, :, None, None]
            result.append(torch.cat((selected, raw[:, :, 2:]), dim=2))
        return tuple(result)
