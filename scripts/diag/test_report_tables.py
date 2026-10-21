import tempfile
from pathlib import Path
import unittest
from scripts.diag import report_tables as t
from scripts.diag.report import format_metric,write_report
from scripts.diag.report_decisions import decide
from scripts.diag.test_report_decisions import evidence,metric

class ReportTableTests(unittest.TestCase):
    def fixture(self):
        e=evidence();m=metric(.25,.1,.4);m['counts']={'pairs':20,'tokens':60};native=[];probe=[]
        for model in ('M5_E1','G3_M5','TRIPLE','G3','G1'):
            for lag in ('A','B','C'):
                native.append(dict(diagnostic='D-a',model=model,layer=24,lag=lag,region='Global',metric='forgetting',**m))
                for feature in ('state','control','memory_specific'):
                    for name in ('cosine','top1','mrr'):probe.append(dict(model=model,layer=24,lag=lag,feature=feature,metric=name,**m))
        for model in ('G3','G1'):
            for layer in range(3,32,4):
                native.append(dict(diagnostic='D-c',model=model,layer=layer,role='combined',
                    category='window8_exact_lag_4_7',metric='mass',**m))
                native.append(dict(diagnostic='D-c',model=model,layer=layer,role='combined',
                    category='bank_lag_8_32',metric='mass',**{**m,'estimate':0.0,'ci95':[0.0,0.0]}))
        repeat=dict(n_episodes=60,success_agreement=1.,metric_agreement=.5,action_sequence_agreement=.5,success_agree_count=60,metric_agree_count=30,action_agree_count=30)
        baselines={model:dict(episodes=4,successes=2,sr=.5) for model in ('M5_E1','G3_M5','TRIPLE','G3','G1')}
        return dict(D_a_metric1=dict(summaries=native,paired_contrasts=[]),D_a_metric1_pair_export=dict(rows=100,models={}),D_a_metric2=dict(summaries=probe),
            D_b=dict(contrasts=[dict(model='G3_M5',contrast='targeted-control',group='C',feature='state',metric='cosine',**m)],difference_in_differences=[dict(feature='state',metric='cosine',**m)],coverage=[]),
            D_b_decision_token_KL=dict(summaries=[dict(model='G3_M5',**m)]),D_e=dict(models={model:dict(metrics={name:{c:m for c in ('score','random','delta')} for name in ('sr','spl','os','ne','forced_stop')},
                forced_stop_counts={'score':2,'random':3},execution_failures={'score':0,'random':1},mcnemar={'p_two_sided':.2},determinism=repeat) for model in ('M5_E1','G3_M5')},interaction=m),
            decisions=decide(e),decision_evidence=e,full_completion=True,recommendation='Test only.',
            provenance=dict(provenance={'baseline_results':baselines},gates={},scope_amendments=[],limitations=[],command_files=[]))
    def test_each_diagnostic_has_measured_table_prediction_and_outcome(self):
        result=self.fixture()
        for name in ('D-a','D-b','D-c','D-d','D-e','D-f'):
            text='\n'.join(t.diagnostic_tables(name,result))
            self.assertIn('Pre-registered prediction',text);self.assertIn('Observed',text);self.assertIn('|---',text)
        text='\n'.join(t.diagnostic_tables('D-a',result));self.assertIn('pairs=20',text);self.assertIn('tokens=60',text)
        self.assertIn('State−control cosine',text);self.assertIn('State retrieval MRR',text)
        text='\n'.join(t.diagnostic_tables('D-e',result));self.assertIn('30/60 episodes',text);self.assertIn('NE (m)',text)
        text='\n'.join(t.diagnostic_tables('D-c',result));self.assertIn('Window8 reference',text)
        self.assertIn('not applicable: no bank tokens',text);self.assertIn('| G3 |',text);self.assertIn('| G1 |',text)
    def test_report_renders_actual_baseline_and_all_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_report(Path(tmp),self.fixture());text=(Path(tmp)/'REPORT.md').read_text()
            self.assertIn('2 / 4',text);self.assertIn('Original full-run results',text)
            for name in ('D-a','D-b','D-c','D-d','D-e','D-f'):self.assertIn('### '+name,text)
            self.assertEqual(text.count('**Observed:**'),6)
    def test_invalid_metrics_and_baseline_ratios_rejected(self):
        with self.assertRaises(ValueError):format_metric(metric(float('nan')))
        b=self.fixture()['provenance']['provenance']['baseline_results'];t.validate_baselines(b)
        self.assertIn('unavailable','\n'.join(t.baseline_table(b)))
        b['G1']['sr']=.75
        with self.assertRaisesRegex(ValueError,'SR differs'):t.validate_baselines(b)
if __name__=='__main__':unittest.main()
