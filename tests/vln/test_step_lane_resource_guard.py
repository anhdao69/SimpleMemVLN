from scripts.vln.run_step_lane_gate import memory_sample


def test_clean_inactive_cache_is_reclaimable(tmp_path):
    (tmp_path / 'memory.current').write_text(str(450 * 2**30))
    (tmp_path / 'memory.stat').write_text(
        f'inactive_file {180 * 2**30}\nfile_dirty {3 * 2**30}\nfile_writeback {2 * 2**30}\n'
    )
    sample = memory_sample(tmp_path)
    assert sample['job_host_memory_gib'] == 450
    assert sample['guard_working_set_gib'] == 275


def test_missing_or_dirty_cache_is_not_subtracted(tmp_path):
    (tmp_path / 'memory.current').write_text(str(10 * 2**30))
    (tmp_path / 'memory.stat').write_text('inactive_file 100\nfile_dirty 200\n')
    assert memory_sample(tmp_path)['guard_working_set_gib'] == 10
    (tmp_path / 'memory.stat').write_text('anon 123\n')
    assert memory_sample(tmp_path)['guard_working_set_gib'] == 10
