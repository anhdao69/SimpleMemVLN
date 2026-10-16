"""G3+A1 frozen online Omega: one single-view call on the union of pair frames.

Descriptors are the unpositioned full-width LN2048 tokens shared by G3 and A1.
G3 regional targets are region means of the SOURCE frames; A1 relation targets
are 32x300 soft relations between READ query cells and SOURCE cells. The
identity omits the unused parent positional-grid bytes (A1's portability repair)
and binds both target constructions' code.
"""
from qwen_vl.research.vggt_omega import OnlineOmegaTeacher,teacher_identity,file_hash


def _g3a1_identity(base):
    import qwen_vl.research.vggt_g3 as regions
    import qwen_vl.models.geoaddress_loss as relations
    unpositioned={k:v for k,v in base.items() if k not in ('position','position_sha256')}
    return dict(base=unpositioned,target='g3a1-unpositioned-LN2048-regions-and-soft-relations-v1',
                adapter_sha256=file_hash(__file__),regions_sha256=file_hash(regions.__file__),
                relations_sha256=file_hash(relations.__file__))


def g3a1_teacher_identity(source,weights):
    return _g3a1_identity(teacher_identity(source,weights))


class OnlineG3A1Teacher(OnlineOmegaTeacher):
    def __init__(self,spec,device):
        base=teacher_identity(spec['teacher_source'],spec['teacher_weights'])
        identity=_g3a1_identity(base)
        if identity!=spec['teacher_identity']:
            raise ValueError('G3A1 teacher identity differs from configured contract')
        super().__init__(dict(spec,teacher_identity=base),device)
        # Parent adds position AFTER pooling/LN; literal zero removes it exactly.
        self.position=self.position.new_zeros(())
        self.identity=identity


def g3a1_targets(descriptors,sources,reads,cells):
    """Return (G3 regions [P,5,2048], A1 relation probabilities [P,32,300])."""
    from qwen_vl.research.vggt_g3 import region_targets
    from qwen_vl.models.geoaddress_loss import teacher_relations
    source=descriptors[sources]
    current=descriptors[reads].gather(1,cells[:,:,None].expand(-1,-1,descriptors.shape[-1]))
    return region_targets(source),teacher_relations(current,source)
