"""R1 uses G3's unpositioned regions from portable online Omega descriptors."""
from qwen_vl.research.geoaddress_teacher import OnlineAddressTeacher,a1_teacher_identity
from qwen_vl.research.vggt_omega import file_hash
from qwen_vl.research.vggt_g3 import region_targets


def r1_teacher_identity(source,weights):
    return dict(descriptors=a1_teacher_identity(source,weights),
                target='r1-unpositioned-LN2048-regions-v1',adapter_sha256=file_hash(__file__),
                regions_sha256=file_hash(__import__('qwen_vl.research.vggt_g3',fromlist=['x']).__file__))


class OnlineR1Teacher(OnlineAddressTeacher):
    def __init__(self,spec,device):
        identity=r1_teacher_identity(spec['teacher_source'],spec['teacher_weights'])
        if identity!=spec['teacher_identity']:raise ValueError('R1 teacher identity differs from contract')
        super().__init__(dict(spec,teacher_identity=identity['descriptors']),device)
        self.identity=identity

    def __call__(self,images):
        if len(images)>8:raise ValueError('R1 permits at most eight unique source images')
        return region_targets(super().__call__(images))
