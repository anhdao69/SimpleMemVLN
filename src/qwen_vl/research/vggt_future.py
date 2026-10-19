"""Frozen paired-frame Omega targets for G1; student inputs stay causal."""
import time
import torch
from qwen_vl.research.vggt_omega import OnlineOmegaTeacher, teacher_identity, file_hash, pool_teacher_tokens


def future_frame_pairs(frames, length, offset):
    """Preserve sampled current frames with an available strictly future view."""
    if type(offset) is not int or offset < 1:
        raise ValueError('Future offset must be a positive integer')
    if type(length) is not int or length < 0:
        raise ValueError('Invalid trajectory length')
    frames = tuple(frames)
    if any(type(t) is not int or not 0 <= t < length for t in frames) or tuple(sorted(set(frames))) != frames:
        raise ValueError('Sampled frames must be ordered unique valid trajectory indices')
    # No wrap, clamping, duplicated terminal image or resampling. Navigation
    # still trains on the complete episode, including its terminal frame.
    return tuple((t, t+offset) for t in frames if t+offset < length)


def future_teacher_identity(source, weights, offset):
    future_frame_pairs((), 0, offset)
    return dict(base=teacher_identity(source, weights),
                target='g1-joint-two-view-current-slot-positioned-v1',
                adapter_sha256=file_hash(__file__), future_offset=offset,
                frame_order=['current', 'future'], output_frame=0,
                terminal_policy='exclude-unavailable-no-resampling')


class OnlineFutureOmegaTeacher(OnlineOmegaTeacher):
    def __init__(self, spec, device):
        identity = future_teacher_identity(spec['teacher_source'], spec['teacher_weights'],
                                           spec['teacher_future_offset'])
        if identity != spec['teacher_identity']:
            raise ValueError('Paired Omega identity differs from configured/resumed contract')
        super().__init__(dict(spec, teacher_identity=identity['base']), device)
        self.identity = identity

    def __call__(self, images):
        if images.ndim != 5 or tuple(images.shape[1:]) != (2, 3, 448, 592) or images.dtype != torch.float32:
            raise ValueError('Paired teacher requires FP32 [pairs,2,3,448,592]')
        if len(images) > 16:
            raise ValueError('G1 samples at most sixteen current/future pairs')
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        if self.residency == 'cpu_between':
            self.model.to(self.device)
        self.model.eval()
        pieces = []
        with torch.inference_mode(), torch.autocast(self.device.type, enabled=False):
            for offset in range(0, len(images), self.batch_size):
                chunk = images[offset:offset+self.batch_size].to(self.device, dtype=torch.float32)
                # Keep S=2: separate batch items cannot exchange geometry.
                outputs, patch_start = self.model.aggregator(chunk)
                tokens = outputs[23]
                if patch_start != 17 or tokens.shape != (len(chunk), 2, 1053, 2048):
                    raise ValueError('Paired Omega layer23 output contract changed')
                # Extract current AFTER joint attention; retain original G1
                # adaptive pooling and LN, never average the two frame slots.
                pieces.append(pool_teacher_tokens(tokens[:, :1], patch_start, (28, 37)))
                del outputs, tokens, chunk
            inferred = torch.cat(pieces) if pieces else torch.empty(0, 300, 2048, device=self.device)
        # Preserve G1's fixed positional target and make ordinary detached
        # tensors which autograd can safely save for the student's backward.
        result = inferred.clone() + self.position
        del inferred, pieces
        if not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite paired Omega target')
        if self.residency == 'cpu_between':
            self.model.cpu()
            torch.cuda.empty_cache()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.last_seconds = time.perf_counter() - start
        return result
