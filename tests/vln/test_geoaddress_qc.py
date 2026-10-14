import pytest
import torch


def test_qc_rejects_cross_image_coupling_but_measures_batch_size_roundoff():
    from scripts.vln.check_geoaddress_teacher import check_batch_independence
    baseline=torch.randn(1,300,2048)
    single=baseline.clone();single[0,0,0]+=1.6e-4
    result=check_batch_independence(baseline,baseline.clone(),single)
    assert result['same_batch_exact'] and result['batch_size_max_abs']>1e-4
    changed=baseline.clone();changed[0,0,0]+=1e-4
    with pytest.raises(AssertionError):check_batch_independence(baseline,changed,single)


def test_profile_revision_works_in_immutable_archive(tmp_path):
    from scripts.vln.profile_vggt_g1 import source_revision
    (tmp_path/'source.commit').write_text('0123456789abcdef\n')
    assert source_revision(tmp_path)==('0123456789abcdef','immutable-source-archive')
