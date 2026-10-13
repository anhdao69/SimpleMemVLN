"""Online G3 regional targets, sharing pinned G1 Omega inference without position."""
from qwen_vl.research.vggt_omega import OnlineOmegaTeacher,teacher_identity,file_hash
from qwen_vl.research.vggt_g3 import region_targets


def g3_teacher_identity(source,weights):
    return dict(base=teacher_identity(source,weights),target='g3-unpositioned-LN2048-regions-v1',
                adapter_sha256=file_hash(__file__),
                regions_sha256=file_hash(__import__('qwen_vl.research.vggt_g3',fromlist=['x']).__file__))


class OnlineG3Teacher(OnlineOmegaTeacher):
    def __init__(self,spec,device):
        identity=g3_teacher_identity(spec['teacher_source'],spec['teacher_weights'])
        if identity!=spec['teacher_identity']:raise ValueError('G3 teacher identity differs from saved contract')
        base=dict(spec,teacher_identity=identity['base'])
        super().__init__(base,device)
        # Parent adds this AFTER pooling/LN. Literal zero removes position without
        # subtracting a previously rounded positional target. No base code change.
        self.position=self.position.new_zeros(())
        self.identity=identity

    def __call__(self,images):
        if len(images)>8:raise ValueError('G3 has at most eight unique source images')
        return region_targets(super().__call__(images))
