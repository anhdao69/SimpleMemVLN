"""Frozen online Omega, trainer-owned and excluded from student/DDP/checkpoints."""
import hashlib
import importlib
from pathlib import Path
import sys
import time
import torch
import torch.nn.functional as F
from qwen_vl.research.vggt_sampling import stable_hash
from qwen_vl.data.vggt_online import PREPROCESS_VERSION,POSITION_VERSION,positional_grid


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def teacher_identity(source,weights):
    source=Path(source).resolve()
    return dict(weights_sha256=file_hash(weights),
        source_sha256=stable_hash([(str(f.relative_to(source)),file_hash(f))
                                  for f in sorted((source/'vggt_omega').rglob('*.py'))]),
        preprocessing=PREPROCESS_VERSION,position=POSITION_VERSION,
        teacher_layer=23,patch_offset=17,teacher_dim=2048,
        compute_dtype='float32',pooling='adaptive-average-15x20-LN2048-eps1e-6',
        position_sha256=hashlib.sha256(positional_grid().numpy().tobytes()).hexdigest())


def pool_teacher_tokens(tokens,patch_start,grid,normalize=True):
    if tokens.ndim!=4 or tokens.shape[1]!=1:
        raise ValueError('Single-view teacher requires [B,1,N,D], never multi-view S')
    h,w=grid;patches=tokens[:,0,patch_start:]
    if patches.shape[1]!=h*w: raise ValueError('Teacher patch grid mismatch')
    pooled=F.adaptive_avg_pool2d(patches.float().transpose(1,2).reshape(-1,tokens.shape[-1],h,w),(15,20))
    pooled=pooled.flatten(2).transpose(1,2).contiguous()
    return F.layer_norm(pooled,(tokens.shape[-1],),eps=1e-6) if normalize else pooled


class OnlineOmegaTeacher:
    def __init__(self,spec,device):
        self.device=torch.device(device); self.spec=dict(spec)
        self.identity=teacher_identity(spec['teacher_source'],spec['teacher_weights'])
        if self.identity!=spec['teacher_identity']:
            raise ValueError('Online teacher identity differs from configured/resumed contract')
        source=Path(spec['teacher_source']).resolve();sys.path.insert(0,str(source))
        module=importlib.import_module('vggt_omega.models')
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise ValueError('A different Omega source is already imported')
        # Teacher construction must not change student/dropout/data RNG states.
        with torch.random.fork_rng(devices=[]):
            model=module.VGGTOmega(enable_camera=False,enable_depth=False,enable_alignment=False)
        state=torch.load(spec['teacher_weights'],map_location='cpu',weights_only=True,mmap=True)
        model.aggregator.load_state_dict({k.removeprefix('aggregator.'):v for k,v in state.items()
                                         if k.startswith('aggregator.')},strict=True)
        self.model=model.eval().requires_grad_(False)
        self.model.aggregator.cached_layer_indices={23}
        self.batch_size=spec.get('teacher_batch_size',8)
        self.residency=spec.get('teacher_residency','resident')
        if self.residency=='resident': self.model.to(self.device)
        self.position=.5*positional_grid().to(self.device)
        self.last_seconds=0.

    def __call__(self,images):
        if images.ndim!=4 or tuple(images.shape[1:])!=(3,448,592) or images.dtype!=torch.float32:
            raise ValueError('Online teacher inputs must be FP32 [F,3,448,592]')
        if len(images)>16: raise ValueError('G1 samples at most16 frames')
        torch.cuda.synchronize(self.device);start=time.perf_counter()
        if self.residency=='cpu_between': self.model.to(self.device)
        self.model.eval()
        pieces=[]
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            for offset in range(0,len(images),self.batch_size):
                chunk=images[offset:offset+self.batch_size].to(self.device,dtype=torch.float32)
                outputs,patch_start=self.model.aggregator(chunk[:,None])
                if patch_start!=17 or outputs[23].shape[2:]!=(1053,2048):
                    raise ValueError('Omega layer23 output contract changed')
                pieces.append(pool_teacher_tokens(outputs[23],patch_start,(28,37)))
                del outputs,chunk
            inferred=torch.cat(pieces) if pieces else torch.empty(0,300,2048,device=self.device)
        # Non-inference tensor: backward through cosine may save detached targets.
        result=inferred.clone()+self.position
        del inferred,pieces
        if not torch.isfinite(result).all(): raise FloatingPointError('Nonfinite online Omega target')
        if self.residency=='cpu_between':
            self.model.cpu()
            torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device);self.last_seconds=time.perf_counter()-start
        return result
