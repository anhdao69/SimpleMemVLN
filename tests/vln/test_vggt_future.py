"""G1 two-view teacher: temporal alignment, ownership and unchanged student inputs."""
import importlib
from types import SimpleNamespace
import pytest
import torch


def module():
    return importlib.import_module('qwen_vl.research.vggt_future')


def test_future_pairs_do_not_wrap_repeat_or_resample_terminal_frames():
    assert module().future_frame_pairs((0,2,4),5,1)==((0,1),(2,3))
    assert module().future_frame_pairs((0,2,4),5,2)==((0,2),(2,4))
    assert module().future_frame_pairs((0,),1,1)==()
    assert module().future_frame_pairs((),7,1)==()
    for offset in (0,-1,True,1.5):
        with pytest.raises(ValueError):module().future_frame_pairs((0,),3,offset)
    for frames in ((-1,),(3,),(1,1),(2,1)):
        with pytest.raises(ValueError):module().future_frame_pairs(frames,3,1)


class JointAggregator(torch.nn.Module):
    """Small cross-view stand-in for the external frozen 1B Omega encoder."""
    def forward(self,images):
        assert images.ndim==5 and images.shape[1]==2
        x=images.mean((2,3,4));c=torch.arange(2048,dtype=images.dtype,device=images.device)
        features=(x[...,None]+1)*torch.sin(c/17)+(x.flip(1)[...,None]+1)*torch.cos(c/31)
        outputs=[None]*24
        outputs[23]=features[:,:,None,:].expand(-1,-1,1053,-1).contiguous()
        return outputs,17


def teacher(batch=2):
    obj=object.__new__(module().OnlineFutureOmegaTeacher)
    obj.device=torch.device('cpu');obj.batch_size=batch;obj.residency='resident'
    obj.model=torch.nn.Module();obj.model.aggregator=JointAggregator()
    from qwen_vl.data.vggt_online import positional_grid
    obj.position=.5*positional_grid()
    return obj


def test_current_slot_after_joint_attention_preserves_original_g1_target_transform():
    obj=teacher();images=torch.zeros(2,2,3,448,592)
    images[0,0]=1;images[0,1]=3;images[1,0]=2;images[1,1]=4
    actual=obj(images)
    from qwen_vl.research.vggt_omega import pool_teacher_tokens
    out,start=obj.model.aggregator(images)
    expected=pool_teacher_tokens(out[23][:,:1],start,(28,37))+obj.position
    wrong=pool_teacher_tokens(out[23][:,1:],start,(28,37))+obj.position
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert not torch.allclose(actual,wrong)
    changed=images.clone();changed[0,1]=9
    target=obj(changed)
    assert not torch.allclose(actual[0],target[0])
    assert torch.equal(actual[1],target[1])


def test_microbatch_consistency_and_backward_safe_detachment():
    images=torch.randn(3,2,3,448,592,requires_grad=True)
    targets=teacher(1)(images)
    # FP32 pooling/LN reductions can vary by a few ULPs with batch shape.
    torch.testing.assert_close(targets,teacher(2)(images),rtol=1e-5,atol=2e-6)
    assert not targets.requires_grad and not targets.is_inference()
    student=torch.randn_like(targets,requires_grad=True)
    torch.nn.functional.cosine_similarity(student,targets,dim=-1).mean().backward()
    assert student.grad.abs().sum()>0 and images.grad is None
    empty=teacher()(torch.empty(0,2,3,448,592))
    assert empty.shape==(0,300,2048) and not empty.is_inference()


@pytest.mark.parametrize('shape,dtype', [((2,3,448,592),torch.float32),((1,1,3,448,592),torch.float32),((1,2,3,448,592),torch.bfloat16),((17,2,3,448,592),torch.float32)])
def test_malformed_teacher_batch_is_rejected(shape,dtype):
    with pytest.raises(ValueError):teacher()(torch.zeros(shape,dtype=dtype))


@pytest.mark.parametrize('length,expected',[(3,[[0,1],[1,2]]),(1,[])])
def test_dataset_alignment_and_future_pixels_never_modify_student_inputs(monkeypatch,length,expected):
    from qwen_vl.data.episode_dataset import EpisodeDataset
    episode={'dataset':'r2r','scene_id':'scene','episode_uid':'id','instruction':'go',
             'steps':[{'rgb_path':f'/trajectory_data/a/{i}.png'} for i in range(length)]}
    ids=torch.full((1,length*302),7);spans=[]
    for i in range(length):
        ids[0,i*302:i*302+300]=99;spans.append((i*302,(i+1)*302))
    encoded={'input_ids':ids,'pixel_values':torch.randn(length,3),
             'step_plan':((0,0),spans),'num_actions':length}
    dataset=object.__new__(EpisodeDataset);dataset.episodes=[episode];dataset.encoded_lengths=[length*302]
    dataset.serializer=SimpleNamespace(image_id=99,config={'runtime':{},'research':{'run_id':'G1',
        'teacher_context':'current_future_v1','teacher_future_offset':1}},encode_episode=lambda ep:dict(encoded))
    def images(paths):
        from pathlib import Path
        return torch.tensor([int(Path(p).stem) for p in paths],dtype=torch.float32)[:,None,None,None].expand(-1,3,448,592)
    monkeypatch.setattr('qwen_vl.data.vggt_online.preprocess_teacher_images',images)
    result=dataset[0]
    assert result['teacher_images'].shape==(len(expected),2,3,448,592)
    assert result['teacher_images'][:,:,0,0,0].tolist()==expected
    assert result['aux_frame_count']==len(expected)
    assert result['aux_visual_positions'].shape==(len(expected),300)
    if expected:assert result['aux_visual_positions'][:,0].tolist()==[0,302]
    for key,value in encoded.items():
        if torch.is_tensor(value):assert torch.equal(value,result[key]),key
        else:assert value==result[key]


def test_future_config_requires_valid_explicit_context():
    from qwen_vl.research.vggt_g1 import parse_g1
    spec=dict(run_id='G1',teacher_source='source',teacher_weights='weights',teacher_identity={'test':True},
              lambda_max=.2,head_lr=1e-4,ramp_updates=771)
    base=dict(memory={'mode':'window8'},model={'output_mode':'qwen_text'})
    for extra in ({'teacher_context':'typo'},{'teacher_future_offset':1},
                  {'teacher_context':'current_future_v1'},
                  {'teacher_context':'current_future_v1','teacher_future_offset':0}):
        with pytest.raises(ValueError):parse_g1(dict(base,research=dict(spec,**extra)))


def test_trainer_consumes_teacher_pixels_before_student_cast(monkeypatch):
    from qwen_vl.train.train_episode import EpisodeTrainer, QwenSFTTrainer
    images=torch.randn(1,2,3,448,592)
    targets=torch.randn(1,300,2048)
    class Teacher:
        last_seconds=.25
        def __call__(self,value):
            assert value is images and value.dtype==torch.float32
            return targets
    def student_prepare(self,inputs):
        assert 'teacher_images' not in inputs and 'aux_targets' not in inputs
        return {k:v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in inputs.items()}
    monkeypatch.setattr(QwenSFTTrainer,'_prepare_inputs',student_prepare)
    trainer=object.__new__(EpisodeTrainer);trainer.online_teacher=Teacher()
    result=trainer._prepare_inputs(dict(teacher_images=images,pixel_values=torch.ones(1),
                                        input_ids=torch.ones(1,dtype=torch.long)))
    assert result['aux_targets'] is targets and targets.dtype==torch.float32
    assert result['pixel_values'].dtype==torch.bfloat16 and 'teacher_images' not in result


def test_identity_pins_pairing_and_rejects_mismatch_before_model_load(tmp_path):
    source=tmp_path/'source';(source/'vggt_omega').mkdir(parents=True)
    (source/'vggt_omega/__init__.py').write_text('# synthetic source')
    weights=tmp_path/'weights';weights.write_bytes(b'identity only')
    identity=module().future_teacher_identity(source,weights,1)
    assert identity!=module().future_teacher_identity(source,weights,2)
    spec=dict(teacher_source=str(source),teacher_weights=str(weights),teacher_future_offset=2,
              teacher_identity=identity)
    with pytest.raises(ValueError,match='identity differs'):
        module().OnlineFutureOmegaTeacher(spec,'cpu')
