"""A1 must bind the teacher it uses, not numerical bytes of unused position."""
import pytest
import torch
from qwen_vl.research import geoaddress_teacher as address
from qwen_vl.research import vggt_omega as omega


@pytest.fixture
def teacher_files(tmp_path):
    source=tmp_path/'source';(source/'vggt_omega').mkdir(parents=True)
    (source/'vggt_omega/model.py').write_text('version=1\n')
    weights=tmp_path/'weights.pt';weights.write_bytes(b'weights')
    return source,weights


def test_a1_identity_ignores_unused_position_roundoff(teacher_files,monkeypatch):
    source,weights=teacher_files
    expected=address.a1_teacher_identity(source,weights)
    grid=omega.positional_grid()
    monkeypatch.setattr(omega,'positional_grid',lambda:grid+1e-7)
    assert address.a1_teacher_identity(source,weights)==expected


def test_a1_still_rejects_changed_teacher_weights(teacher_files):
    source,weights=teacher_files
    identity=address.a1_teacher_identity(source,weights)
    weights.write_bytes(b'changed weights')
    with pytest.raises(ValueError,match='identity'):
        address.OnlineAddressTeacher(dict(teacher_source=source,teacher_weights=weights,
                                         teacher_identity=identity),'cpu')


def test_a1_initialization_keeps_strict_base_check_and_zero_position(teacher_files,monkeypatch):
    source,weights=teacher_files
    identity=address.a1_teacher_identity(source,weights)
    grid=omega.positional_grid()
    monkeypatch.setattr(omega,'positional_grid',lambda:grid+1e-7)
    calls=[]
    def base_init(self,spec,device):
        assert spec['teacher_identity']==omega.teacher_identity(source,weights)
        calls.append(spec)
        self.position=torch.ones(300,2048)
    monkeypatch.setattr(omega.OnlineOmegaTeacher,'__init__',base_init)
    teacher=address.OnlineAddressTeacher(dict(teacher_source=source,teacher_weights=weights,
                                            teacher_identity=identity),'cpu')
    assert len(calls)==1 and teacher.position.ndim==0 and teacher.position.item()==0
    assert teacher.identity==identity
