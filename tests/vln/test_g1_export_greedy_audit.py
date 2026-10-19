import pytest
from scripts.vln.check_vggt_export import greedy_observation

class Session:
    def __init__(self,error=None):self.error=error
    def reset(self,uid,instruction):self.reset_args=(uid,instruction)
    def observe(self,uid,step,rgb):
        if self.error:raise self.error
        return {'response_token_ids':[1,2]}

def test_valid_and_reset():
    session=Session()
    assert greedy_observation(session,'a','go',None)==dict(response_token_ids=[1,2],error=None)
    assert session.reset_args==('a','go')

def test_smoke_only_known_error_is_recorded():
    message='Response limit reached without assistant terminator'
    session=Session(ValueError(message))
    with pytest.raises(ValueError):greedy_observation(session,'a','go',None)
    assert greedy_observation(session,'a','go',None,allow_untrained=True)==dict(response_token_ids=None,error=message)

@pytest.mark.parametrize('error',[ValueError('Nonfinite scores'),RuntimeError('CUDA failure')])
def test_unrelated_failures_remain_fatal(error):
    with pytest.raises(type(error),match=str(error)):
        greedy_observation(Session(error),'a','go',None,allow_untrained=True)
