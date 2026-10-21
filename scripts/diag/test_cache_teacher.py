import unittest
import torch
from scripts.diag.cache_teacher import collect_targets
from scripts.diag import cache_teacher


class CacheTeacherTests(unittest.TestCase):
    def test_batch_limit_order_and_detached_fp16(self):
        calls = []
        def preprocess(paths):
            return torch.tensor(paths, dtype=torch.float32)
        def teacher(batch):
            calls.append(len(batch))
            return batch[:, None, None].expand(-1, 5, 2048).clone()
        result = collect_targets(list(range(19)), teacher, preprocess)
        self.assertEqual(calls, [8, 8, 3])
        self.assertEqual(result.shape, (19, 5, 2048))
        self.assertEqual(result.dtype, torch.float16)
        self.assertFalse(result.requires_grad)
        self.assertTrue(torch.equal(result[:, 0, 0].float(), torch.arange(19).float()))

    def test_reject_nonfinite_wrong_shape_or_precision(self):
        for value in [torch.zeros(1, 4, 2048), torch.full((1, 5, 2048), float('nan')),
                      torch.zeros(1, 5, 2048, dtype=torch.float16)]:
            with self.assertRaises(ValueError):
                collect_targets(['x'], lambda batch: value, lambda paths: paths)

    def test_half_overflow_rejected(self):
        with self.assertRaises(ValueError):
            collect_targets(['x'],lambda batch:torch.full((1,5,2048),1e10),lambda paths:paths)

    def test_frozen_panel_rejects_partial_and_truncated_inputs(self):
        panel={'episodes':[dict(episode_id=str(i),scene='scene',gt_action_count=2) for i in range(240)]}
        episodes=[dict(episode_id=str(i),episode_uid=str(i),scene_id='mp3d/scene/scene.glb',steps=[{},{}]) for i in range(240)]
        cache_teacher.validate_panel(episodes,panel)
        with self.assertRaises(ValueError):cache_teacher.validate_panel(episodes[:-1],panel)
        episodes[0]['steps'].pop()
        with self.assertRaises(ValueError):cache_teacher.validate_panel(episodes,panel)

    def test_empty_episode_rejected(self):
        with self.assertRaises(ValueError):
            collect_targets([], None, None)


if __name__ == '__main__':
    unittest.main()
