import json
import numpy as np
import pytest
import torch
from qwen_vl.research.vggt_sampling import sample_frames, collection_trajectory_key, sampled_visual_positions
from qwen_vl.data.vggt_online import preprocess_teacher_images, positional_grid
from qwen_vl.research.vggt_omega import pool_teacher_tokens

@pytest.mark.parametrize('length', [1,7,8,9,16,627])
def test_stratified_sampling(length):
    frames = sample_frames('path', length, 429)
    assert frames == sample_frames('path', length, 429)
    assert len(set(frames)) == min(16,length)
    assert list(frames) == sorted(frames)
    for j, frame in enumerate(frames):
        assert j*length//len(frames) <= frame < (j+1)*length//len(frames)

def test_instruction_does_not_change_trajectory():
    ep = dict(dataset='r2r', scene_id='scene', instruction='a', steps=[dict(rgb_path='/data/trajectory_data/R2R/train/1/a.png')])
    assert collection_trajectory_key(ep) == collection_trajectory_key(dict(ep, instruction='b', episode_uid='different'))

def test_visual_positions_exclude_text_and_future_groups():
    ids=torch.tensor([[9,9,7,7,7,4,9,7,7,7,5]])
    assert sampled_visual_positions(ids,7,[(2,6),(6,11)],(1,),tokens_per_frame=3).tolist()==[[7,8,9]]
    with pytest.raises(ValueError):
        sampled_visual_positions(ids,7,[(2,6),(6,11)],(1,),tokens_per_frame=4)

def test_pool_uses_returned_special_offset_and_row_major():
    grid=torch.arange(15*20).reshape(1,1,300,1).expand(1,1,300,4).float()
    special=torch.full((1,1,17,4),-100.)
    result=pool_teacher_tokens(torch.cat([special,grid],2),17,(15,20),normalize=False)
    assert torch.equal(result,grid[:,0])
    with pytest.raises(ValueError):
        pool_teacher_tokens(torch.cat([special,grid],2).expand(1,2,-1,-1),17,(15,20))


def test_online_preprocess_matches_official_loader(tmp_path):
    from PIL import Image
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    pixels=np.random.default_rng(429).integers(0,256,(480,640,3),dtype=np.uint8)
    path=tmp_path/'rgb.png';Image.fromarray(pixels).save(path)
    actual=preprocess_teacher_images([str(path)])
    expected=load_and_preprocess_images([str(path)],image_resolution=512)
    assert torch.equal(actual,expected) and actual.dtype==torch.float32


def test_online_position_grid_full_teacher_dimension():
    e=positional_grid()
    assert e.shape==(300,2048)
    torch.testing.assert_close(e.square().mean(),torch.tensor(1.))


def test_teacher_identity_binds_weights_external_source_and_local_adapter(tmp_path):
    from qwen_vl.research.vggt_omega import teacher_identity,file_hash
    import qwen_vl.research.vggt_omega as adapter
    import qwen_vl.data.vggt_online as preprocessing
    source=tmp_path/'source';(source/'vggt_omega').mkdir(parents=True)
    code=source/'vggt_omega/model.py';code.write_text('version=1\n')
    weights=tmp_path/'weights.pt';weights.write_bytes(b'first')
    first=teacher_identity(source,weights)
    assert first['adapter_sha256']==file_hash(adapter.__file__)
    assert first['preprocessing_code_sha256']==file_hash(preprocessing.__file__)
    weights.write_bytes(b'second')
    assert teacher_identity(source,weights)['weights_sha256']!=first['weights_sha256']
    code.write_text('version=2\n')
    assert teacher_identity(source,weights)['source_sha256']!=first['source_sha256']


def test_actual_student_processor_and_merger_preserve_row_major_cells():
    import os
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionPatchMerger
    processor=AutoProcessor.from_pretrained(os.environ['VLN_MODEL_PATH'])
    # Every32x32 block is exactly one student merged token, in row-major order.
    y,x=np.meshgrid(np.arange(15),np.arange(20),indexing='ij')
    colors=np.stack((x*11,y*17,(x+y)*7),-1).astype(np.uint8)
    pixels=np.repeat(np.repeat(colors,32,0),32,1)
    encoded=processor.image_processor(images=[Image.fromarray(pixels)],return_tensors='pt')
    assert encoded['image_grid_thw'].tolist()==[[1,30,40]]
    means=encoded['pixel_values'].reshape(300,4,3,2,16,16).mean((1,3,4,5))
    expected=torch.from_numpy(colors.reshape(300,3)).float()/255
    expected=(expected-torch.tensor(processor.image_processor.image_mean))/torch.tensor(processor.image_processor.image_std)
    torch.testing.assert_close(means,expected,atol=2e-6,rtol=2e-6)
    cfg=Qwen3_5VisionConfig(hidden_size=4,out_hidden_size=4,spatial_merge_size=2)
    merger=Qwen3_5VisionPatchMerger(cfg).eval()
    features=torch.randn(1200,4)
    with torch.no_grad():
        together=merger(features)
        individually=torch.cat([merger(row) for row in features.reshape(300,4,4)])
    torch.testing.assert_close(together,individually,atol=1e-6,rtol=1e-5)
