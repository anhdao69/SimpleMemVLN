import unittest
from scripts.diag.report_decisions import decide

def metric(v,lo=None,hi=None):return {'estimate':v,'ci95':[v if lo is None else lo,v if hi is None else hi],'n_episodes':240}
def evidence():
    return {'r1_b':metric(.1,.01,.2),'r1_c':metric(-.1,-.2,-.01),
            'r1_probe_b':metric(-.01),'r1_probe_c':metric(.01),
            'r2_target':metric(-.02,-.03,-.01),'r2_attention':metric(.001,.0001,.002),
            'r3_margins':{str(l):metric(-.1,-.2,-.01) for l in (2,6,10,14,18,22)},
            'r3_interaction':metric(.015,-.01,.04),'r4_mass':metric(-.01,-.02,-.001),
            'r5_overlap':{f'{m}:{l}':metric(.995) for m in ('G3_M5','TRIPLE') for l in range(3,32,4)}}
class Decisions(unittest.TestCase):
    def test_boundaries_and_raw_probe_direction(self):
        e=evidence();d=decide(e)
        self.assertEqual([d[k]['status'] for k in ('R1','R2','R3','R4','R5')],['Supported']*4+['Refuted'])
        e['r1_probe_b']=metric(.01);self.assertEqual(decide(e)['R1']['status'],'Inconclusive')
        e['r5_overlap']['TRIPLE:31']=metric(.98);self.assertEqual(decide(e)['R5']['status'],'Inconclusive')
        e['r5_overlap']['TRIPLE:31']=metric(.9799);self.assertEqual(decide(e)['R5']['status'],'Supported')
    def test_missing_never_refuted(self):
        d=decide({});self.assertTrue(all(x['status']=='Inconclusive' for x in d.values()))
        e=evidence();del e['r5_overlap']['G3_M5:3'];self.assertEqual(decide(e)['R5']['status'],'Inconclusive')
    def test_five_layers_and_interaction_threshold(self):
        e=evidence();e['r3_margins']['22']=metric(0,-.1,.1)
        self.assertEqual(decide(e)['R3']['status'],'Supported')
        e['r3_interaction']=metric(.014999);self.assertEqual(decide(e)['R3']['status'],'Inconclusive')
    def test_zero_touch_not_excluding_zero(self):
        e=evidence();e['r1_b']=metric(.1,0,.2);self.assertEqual(decide(e)['R1']['status'],'Inconclusive')
        e['r2_target']=metric(0,-.1,.1);e['r2_attention']=metric(0,-.1,.1)
        self.assertEqual(decide(e)['R2']['status'],'Refuted')
if __name__=='__main__':unittest.main()
