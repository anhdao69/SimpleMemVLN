import csv
import json
from pathlib import Path
import tempfile
import unittest
import importlib.util
from scripts.diag.report_figures import spatial_maps,lag_position

class FigureTests(unittest.TestCase):
    def test_spatial_map_equal_episode_weight_not_token_pooled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'selection.csv';a=[[0]*20 for _ in range(15)];b=[[0]*20 for _ in range(15)];a[0][0]=100;b[0][1]=1
            with path.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=['model','scorer_layer','episode_id','spatial_histogram_json']);writer.writeheader()
                for eid,grid in [('a',a),('b',b)]:writer.writerow(dict(model='M5_E1',scorer_layer=2,episode_id=eid,spatial_histogram_json=json.dumps(grid)))
            result=spatial_maps(path)[0]
            self.assertEqual(result['n_token_visits'],101);self.assertEqual(result['n_episodes'],2)
            self.assertEqual(result['mean_episode_occupancy_share'][0][:2],[.5,.5])
    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Plotting environment required')
    def test_six_png_groups_keep_source_numbers(self):
        from scripts.diag.report_figures import build_figures
        metric=dict(estimate=.5,ci95=[.4,.6],n_episodes=240)
        result=dict(D_a_metric1={'summaries':[]},D_a_metric2={'summaries':[]},D_b={'contrasts':[]},
            D_e=dict(models={m:{'metrics':{'sr':{c:metric for c in ('score','random','delta')}}} for m in ('M5_E1','G3_M5')},interaction=metric))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);csvfile=root/'selection.csv'
            csvfile.write_text('model,scorer_layer,episode_id,spatial_histogram_json\n')
            figures=build_figures(result,root,csvfile)
            self.assertEqual(set(figures),{'D-a','D-b','D-c','D-d','D-e','D-f'})
            self.assertEqual(figures['D-e']['data'][0]['estimate'],.5)
            for entry in figures.values():self.assertGreater((root/entry['file']).stat().st_size,1000)

    def test_registered_bucket_axis(self):
        self.assertEqual(lag_position('13-16'),13);self.assertIsNone(lag_position('C'))

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Plotting environment required')
    def test_single_group_confidence_interval_has_visible_line(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from scripts.diag.report_figures import series
        fig,ax=plt.subplots()
        try:
            series(ax,[dict(model='G3_M5',x=2,estimate=.1,ci95=[-.2,.3])],'x')
            bars=[c for c in ax.collections if isinstance(c,LineCollection)]
            self.assertEqual(len(bars),1)
            self.assertEqual(bars[0].get_segments()[0].tolist(),[[2.,-.2],[2.,.3]])
        finally:plt.close(fig)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Plotting environment required')
    def test_completed_figures_do_not_need_or_invent_selector_results(self):
        from scripts.diag.report_figures import build_figures
        row=dict(diagnostic='D-a',model='M5_E1',layer=24,region='Global',metric='epsilon',
                 lag='1',estimate=.3,ci95=[.2,.4],n_episodes=240)
        result=dict(D_a_metric1={'summaries':[row]},D_a_metric2={'summaries':[]},D_b={'contrasts':[]})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);csvfile=root/'selection.csv'
            csvfile.write_text('model,scorer_layer,episode_id,spatial_histogram_json\n')
            figures=build_figures(result,root,csvfile,include_selector=False)
            self.assertEqual(set(figures),{'D-a','D-b','D-c','D-d','D-f'})
            self.assertEqual(figures['D-a']['data'],[row])
            self.assertFalse((root/'figures/D-e.png').exists())
            self.assertNotIn('D_e',result)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Plotting environment required')
    def test_final_figures_require_selector_before_writing(self):
        from scripts.diag.report_figures import build_figures
        result=dict(D_a_metric1={'summaries':[]},D_a_metric2={'summaries':[]},D_b={'contrasts':[]})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);csvfile=root/'selection.csv'
            csvfile.write_text('model,scorer_layer,episode_id,spatial_histogram_json\n')
            with self.assertRaises(KeyError):build_figures(result,root,csvfile)
            self.assertFalse((root/'figures').exists())
if __name__=='__main__':unittest.main()
