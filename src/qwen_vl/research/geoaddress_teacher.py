"""A1 frozen online Omega descriptors: full width, single-view, no PCA/position."""
from qwen_vl.research.vggt_omega import OnlineOmegaTeacher,teacher_identity,file_hash


def a1_teacher_identity(source,weights):
    return dict(base=teacher_identity(source,weights),target='a1-unpositioned-LN2048-soft-relations-v1',
                adapter_sha256=file_hash(__file__))


class OnlineAddressTeacher(OnlineOmegaTeacher):
    def __init__(self,spec,device):
        identity=a1_teacher_identity(spec['teacher_source'],spec['teacher_weights'])
        if identity!=spec['teacher_identity']:
            raise ValueError('A1 teacher identity differs from configured contract')
        super().__init__(dict(spec,teacher_identity=identity['base']),device)
        self.position=self.position.new_zeros(())
        self.identity=identity
