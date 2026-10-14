import torch


def test_final_reference_strips_research_metadata_for_all_arms():
    from qwen_vl.research.vggt_g1 import inference_sample
    base={'input_ids':torch.ones(1,3,dtype=torch.long),'episode_uid':'ep','num_actions':1}
    keys=('teacher_images','aux_visual_positions','aux_frame_count','aux_targets','aux_pair_count',
          'aux_anchors','aux_lags','aux_source_indices','aux_query_positions','aux_key_positions',
          'aux_teacher_sources','aux_teacher_reads','aux_query_cells','update_aux_count')
    sample=dict(base,**{k:torch.tensor(1) for k in keys})
    result=inference_sample(sample,'cpu')
    assert result.keys()==base.keys()
    assert result['episode_uid']=='ep' and torch.equal(result['input_ids'],base['input_ids'])
