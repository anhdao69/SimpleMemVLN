import json
import numpy as np
import pytest
import torch
from qwen_vl.research.vggt_sampling import sample_frames, physical_trajectory_key, sampled_visual_positions
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
    assert physical_trajectory_key(ep) == physical_trajectory_key(dict(ep, instruction='b', episode_uid='different'))

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
