import math
import torch
from step_lane_test_utils import reference_kernel


def test_key_by_value_orientation_and_read_after_write():
    q = torch.tensor([[[[0., 1., 0.]]]])
    k = torch.tensor([[[[1., 2., 0.]]]])
    v = torch.tensor([[[[1., 2., 3., 4., 5.]]]])
    out, final = reference_kernel(q, k, v, torch.zeros(1, 1, 1), torch.full((1, 1, 1), .5),
                                  output_final_state=True)
    assert final.shape == (1, 1, 3, 5)
    torch.testing.assert_close(final[0, 0], torch.tensor([[.5, 1., 1.5, 2., 2.5],
                                                        [1., 2., 3., 4., 5.], [0., 0., 0., 0., 0.]]))
    torch.testing.assert_close(out[0, 0, 0], torch.tensor([1., 2., 3., 4., 5.]) / math.sqrt(3))


def test_nonwriter_state_identity_but_query_dependent_reads():
    state = torch.arange(15.).reshape(1, 1, 3, 5)
    q = torch.tensor([[[[1., 0., 0.]], [[0., 1., 0.]]]])
    k, v = torch.ones_like(q), torch.full((1, 2, 1, 5), 100.)
    out, final = reference_kernel(q, k, v, torch.zeros(1, 2, 1), torch.zeros(1, 2, 1),
                                  initial_state=state, output_final_state=True)
    assert torch.equal(final, state)
    assert not torch.equal(out[:, 0], out[:, 1])
