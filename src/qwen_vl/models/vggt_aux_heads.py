"""Training-only G1 head; no teacher and no changes to deployed policy."""
import torch
from torch import nn
from torch.nn import functional as F


class CurrentSpatialProjector(nn.Module):
    def __init__(self, hidden_size=2560, intermediate_size=4096, target_size=2048, seed=429):
        super().__init__()
        # CPU construction inside fork avoids changing the backbone/data RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.norm=nn.LayerNorm(hidden_size,elementwise_affine=True,eps=1e-5)
            self.fc1=nn.Linear(hidden_size,intermediate_size)
            self.fc2=nn.Linear(intermediate_size,target_size)
            for layer in (self.fc1,self.fc2):
                nn.init.normal_(layer.weight,std=.02); nn.init.zeros_(layer.bias)

    def forward(self, hidden, target):
        if hidden.ndim != 3 or target.shape != (*hidden.shape[:2],self.fc2.out_features):
            raise ValueError('Alignment hidden/target dimensions disagree')
        if hidden.shape[0] == 0:
            return sum(p.float().sum()*0 for p in self.parameters()) + hidden.float().sum()*0
        pred=self.fc2(F.gelu(self.fc1(self.norm(hidden))))
        pred=F.normalize(pred.float(),dim=-1,eps=1e-6)
        truth=F.normalize(target.detach().float(),dim=-1,eps=1e-6)
        return (1-(pred*truth).sum(-1)).mean(-1).sum()
