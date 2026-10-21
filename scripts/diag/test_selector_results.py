import unittest
import json
import tempfile
from pathlib import Path
from scripts.diag.selector_results import effective_metric, paired_rows, determinism, read_journal


def row(eid,success,failed=False):
    return dict(episode_id=eid,episode_uid='uid:'+eid,scene_id='scene',instruction='walk',failed=failed,
        metrics={'success':success,'spl':.5*success,'oracle_success':1.,'distance_to_goal':2.},forced_stop=False,actions=[])


class SelectorResultTests(unittest.TestCase):
    def test_failed_sr_zero_but_valid_partial_ne_retained(self):
        r=row('a',1.,True)
        self.assertEqual(effective_metric(r,'sr'),0.)
        self.assertEqual(effective_metric(r,'ne'),2.)

    def test_worker_rejects_extra_episodes_but_baseline_can_contain_full_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'journal.jsonl'
            path.write_text('\n'.join(json.dumps(row(e,1)) for e in ('a','b')))
            self.assertEqual(set(read_journal(path,{'a'})),{'a'})
            with self.assertRaises(ValueError):read_journal(path,{'a'},exact=True)

    def test_pair_identity_cannot_be_silently_joined(self):
        a={'a':row('a',1)};b={'a':row('a',0)}
        paired=paired_rows(a,b,'sr');self.assertEqual(paired[0]['right']-paired[0]['left'],-1)
        b['a']['episode_uid']='different'
        with self.assertRaises(ValueError):paired_rows(a,b,'sr')

    def test_determinism_distinguishes_success_and_numeric_metrics(self):
        a={'a':row('a',1)};b={'a':row('a',1)}
        b['a']['metrics']['distance_to_goal']=2.1
        result=determinism(a,b)
        self.assertEqual(result['success_agreement'],1.)
        self.assertEqual(result['metric_agreement'],0.)


if __name__=='__main__':unittest.main()
