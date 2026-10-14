"""A1 frozen online Omega descriptors: full width, single-view, no PCA/position."""
from qwen_vl.research.vggt_omega import OnlineOmegaTeacher,teacher_identity,file_hash


def _address_identity(base):
    # A1 replaces the parent's positional grid with literal zero. Its numerical
    # bytes can vary with CPU math kernels and are not part of A1's target.
    # Keep all weight/source/preprocessing hashes; only omit unused position.
    unpositioned={k:v for k,v in base.items() if k not in ('position','position_sha256')}
    return dict(base=unpositioned,target='a1-unpositioned-LN2048-soft-relations-v1',
                adapter_sha256=file_hash(__file__))


def a1_teacher_identity(source,weights):
    return _address_identity(teacher_identity(source,weights))


class OnlineAddressTeacher(OnlineOmegaTeacher):
    def __init__(self,spec,device):
        base=teacher_identity(spec['teacher_source'],spec['teacher_weights'])
        identity=_address_identity(base)
        if identity!=spec['teacher_identity']:
            raise ValueError('A1 teacher identity differs from configured contract')
        # Outer validation binds every used target component. Pass the complete
        # locally computed base identity to the unchanged strict parent check.
        super().__init__(dict(spec,teacher_identity=base),device)
        self.position=self.position.new_zeros(())
        self.identity=identity
