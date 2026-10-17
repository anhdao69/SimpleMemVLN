"""Training-only native-state queries and decoder; no teacher-conditioned query."""
import torch
from torch import nn
from torch.nn import functional as F


class NativeStateHead(nn.Module):
    def __init__(self,heads=32,key_dim=128,value_dim=128,query_hidden=512,
                 decoder_hidden=1024,target_dim=2048,seed=429):
        super().__init__()
        self.heads,self.key_dim,self.value_dim=heads,key_dim,value_dim
        # Isolated CPU generator: preserve student RNG on every device.
        with torch.random.fork_rng(devices=[]):
            self.region_embedding=nn.Parameter(torch.empty(5,32))
            self.query=nn.Sequential(nn.LayerNorm(heads*key_dim+64,elementwise_affine=False,eps=1e-6),
                nn.Linear(heads*key_dim+64,query_hidden),nn.SiLU(),nn.Linear(query_hidden,heads*key_dim))
            self.decoder=nn.Sequential(nn.Linear(heads*value_dim,decoder_hidden),nn.GELU(),
                                       nn.Linear(decoder_hidden,target_dim))
        generator=torch.Generator(device='cpu').manual_seed(seed)
        with torch.no_grad():
            for name,p in self.named_parameters():
                if name.endswith('bias'):p.zero_()
                else:p.normal_(0,.02,generator=generator)

    def queries(self,anchor_q,lags):
        if anchor_q.shape!=(len(lags),self.heads,self.key_dim):raise ValueError('G3 context shape')
        context=anchor_q.detach().float()
        # Same additive-epsilon L2 rule as pinned native FLA; cast back as FLA does.
        context=(context*torch.rsqrt(context.square().sum(-1,keepdim=True)+1e-6)).to(anchor_q.dtype)
        context=context.flatten(1)[:,None].expand(-1,5,-1)
        frequencies=10000**(-torch.arange(16,device=anchor_q.device,dtype=torch.float32)/16)
        phase=lags.float()[:,None]*frequencies
        lag=torch.cat((phase.sin(),phase.cos()),-1)[:,None].expand(-1,5,-1)
        region=self.region_embedding[None].expand(len(lags),-1,-1)
        dtype=self.region_embedding.dtype
        features=torch.cat((context.to(dtype),region,lag.to(dtype)),-1)
        return self.query(features).reshape(-1,self.heads,self.key_dim)

    def loss(self,reads,targets):
        if reads.shape!=(len(targets),5,self.heads,self.value_dim):raise ValueError('G3 read shape')
        values=reads.float()
        values=values*torch.rsqrt(values.square().mean(-1,keepdim=True)+1e-6)
        pred=self.decoder(values.flatten(2).to(self.region_embedding.dtype)).float()
        if pred.shape!=targets.shape:raise ValueError('G3 target shape')
        distance=1-(F.normalize(pred,dim=-1,eps=1e-6)*F.normalize(targets.detach().float(),dim=-1,eps=1e-6)).sum(-1)
        return distance.mean(-1).sum()

    def empty_loss(self,hidden):
        return hidden.sum()*0+sum(p.sum()*0 for p in self.parameters())


def build_state_heads(layers, **kwargs):
    """Independent, identically initialized heads; never consume backbone/data RNG."""
    return nn.ModuleDict({str(index):NativeStateHead(**kwargs) for index in layers})
