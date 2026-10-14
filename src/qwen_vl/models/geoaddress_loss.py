"""A1 direct native Q/K relation KL against detached online Omega descriptors.

Different feature widths need no converter: both sides form 32x300 similarity
probabilities. No PCA, learned projection, added position, or hard match labels.
"""
import torch
from torch.nn import functional as F


def native_normalize(x):
    x=x.float()
    return x*torch.rsqrt(x.square().sum(-1,keepdim=True)+1e-6)


def teacher_relations(current,source):
    if (current.ndim!=3 or source.ndim!=3 or current.shape[0]!=source.shape[0]
            or current.shape[1]!=32 or source.shape[1]!=300 or current.shape[2]!=source.shape[2]):
        raise ValueError('Teacher relation expects [P,32,D] queries and [P,300,D] sources')
    with torch.no_grad():
        current,source=current.detach().float(),source.detach().float()
        for value in (current,source):
            if not torch.isfinite(value).all() or (value.square().sum(-1)<=1e-12).any():
                raise FloatingPointError('Nonfinite or zero-norm Omega descriptor')
        q,k=F.normalize(current,dim=-1,eps=1e-6),F.normalize(source,dim=-1,eps=1e-6)
        return (torch.bmm(q,k.transpose(1,2))/.1).softmax(-1)


def relation_kl_sum(q,k,teacher_probs):
    if (q.ndim!=4 or k.ndim!=4 or q.shape[1:3]!=(32,2) or k.shape[1:3]!=(300,2)
            or q.shape[0]!=k.shape[0] or q.shape[3]!=k.shape[3]
            or teacher_probs.shape!=(q.shape[0],32,300)):
        raise ValueError('A1 relation head/index dimensions disagree')
    truth=teacher_probs.detach().float()
    if (not torch.isfinite(truth).all() or (truth<0).any()
            or not torch.allclose(truth.sum(-1),torch.ones_like(truth[:,:,0]),atol=1e-5,rtol=1e-5)):
        raise ValueError('Invalid teacher relation probabilities')
    logits=torch.einsum('pihd,pjhd->phij',q.float(),k.float())/.1
    kl=truth[:,None]*(truth.clamp_min(1e-30).log()[:,None]-logits.log_softmax(-1))
    return kl.sum(-1).mean((1,2)).sum()


def address_relation_loss(q,k,teacher_probs):
    return relation_kl_sum(native_normalize(q),native_normalize(k),teacher_probs)
