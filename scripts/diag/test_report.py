import csv
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.diag import report
from scripts.diag.recall import sampled_pairs,lag_group,lag_bucket
from scripts.diag.cache_teacher import sha256

class ReportTests(unittest.TestCase):
    def test_large_csv_exports_reuse_hash_verified_bundle_and_copy_new_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=root/'completed_exports';bundle.mkdir();exports=root/'stage'/'exports';exports.mkdir(parents=True)
            source=root/'capture.csv';source.write_bytes(b'a,b\n1,2\n')
            part=bundle/'capture.csv.part-000.gz';part.write_bytes(gzip.compress(source.read_bytes(),mtime=0))
            part_record=dict(path=part.name,bytes=part.stat().st_size,sha256=hashlib.sha256(part.read_bytes()).hexdigest())
            source_record=dict(source=str(source),relative_source='capture.csv',source_bytes=source.stat().st_size,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),csv_records_excluding_header=1,
                parts=[part_record])
            manifest=bundle/'manifest.json';manifest.write_text(json.dumps(dict(status='verified',sources=[source_record])))
            extra=root/'episodes.csv';extra.write_text('episode_id,sr\n1,1\n')
            inventory=report.stage_csv_exports([source,extra],exports,bundle)
            packaged=next(x for x in inventory if x['kind']=='chunked_gzip_csv')
            copied=next(x for x in inventory if x['kind']=='copied_csv')
            self.assertEqual(packaged['path'],'completed_exports/capture.csv')
            self.assertFalse((exports/'capture.csv').exists())
            self.assertTrue((root/'stage'/copied['path']).is_file())
            part.write_bytes(part.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(ValueError,'part changed'):
                report.stage_csv_exports([source,extra],exports,bundle)

    def test_unbundled_native_pair_csv_is_split_and_round_trip_verified(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(report,'CHUNK_BYTES',12):
            root=Path(tmp);bundle=root/'completed_exports';bundle.mkdir();exports=root/'stage'/'exports';exports.mkdir(parents=True)
            (bundle/'manifest.json').write_text(json.dumps(dict(status='verified',sources=[])))
            source=root/'native_recall_pairs.csv';payload=b'model,episode\n' + b'M5_E1,e1\n'*8
            source.write_bytes(payload)
            inventory=report.stage_csv_exports([source],exports,bundle)
            item=inventory[0]
            self.assertEqual(item['kind'],'chunked_gzip_csv')
            self.assertEqual(item['rows'],8)
            reconstructed=b''.join(gzip.open(bundle/part['path'],'rb').read() for part in item['parts'])
            self.assertEqual(reconstructed,payload)
            self.assertEqual(report.sha256(bundle/'native_recall_pairs.manifest.json'),item['bundle_manifest_sha256'])
            again=report.stage_csv_exports([source],exports,bundle)
            self.assertEqual(again[0]['source_sha256'],item['source_sha256'])

    def test_unbundled_large_ablation_csvs_with_same_basename_get_distinct_verified_bundles(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(report,'CHUNK_BYTES',16):
            root=Path(tmp);bundle=root/'completed_exports';bundle.mkdir();exports=root/'stage'/'exports';exports.mkdir(parents=True)
            (bundle/'manifest.json').write_text(json.dumps(dict(status='verified',sources=[])))
            sources=[];payloads={}
            for model in ('M5_E1','G3_M5'):
                source=root/'ablation_probe_results'/model/'pairs.csv';source.parent.mkdir(parents=True)
                payload=(f'model,episode\n{model},'+'episode-id-with-a-long-payload\n')*5
                source.write_bytes(payload.encode());sources.append(source);payloads[source]=payload.encode()
            inventory=report.stage_csv_exports(sources,exports,bundle)
            self.assertEqual(len(inventory),2)
            self.assertEqual({item['kind'] for item in inventory},{'chunked_gzip_csv'})
            self.assertEqual(len({item['path'] for item in inventory}),2)
            for item,source in zip(inventory,sources):
                reconstructed=b''.join(gzip.open(root/'stage'/part['path'],'rb').read() for part in item['parts'])
                self.assertEqual(reconstructed,payloads[source])
                self.assertEqual(item['source_sha256'],hashlib.sha256(payloads[source]).hexdigest())
                manifest=root/'stage'/item['bundle_manifest']
                self.assertTrue(manifest.is_file())
                self.assertEqual(report.sha256(manifest),item['bundle_manifest_sha256'])

    def fixture(self,root):
        panel=root/'panel.json';episodes=[dict(episode_id=e,scene='s',gt_action_count=n) for e,n in [('a',10),('b',12)]]
        panel.write_text(json.dumps({'episodes':episodes}))
        for group in ('A','B','C'):
            for feature,layer in [('control',24),('state',22),('state',24),('state',26)]:
                folder=root/'probes'/'M5_E1'/group/f'{feature}_layer_{layer}';folder.mkdir(parents=True)
                for i in (0,1):
                    (folder/f'fold_{i}.pt').write_bytes(b'trusted fixture')
                    (folder/f'fold_{i}.json').write_text(json.dumps(dict(model_file=f'fold_{i}.pt',model_sha256=sha256(folder/f'fold_{i}.pt'),
                        test_scenes=['full/s'] if i==0 else ['other'],train_scenes=['other'] if i==0 else ['full/s'])))
                rr=[]
                for e in episodes:
                    for t,s in sampled_pairs(e['gt_action_count']):
                        if lag_group(t-s)!=group:continue
                        value=(0 if e['episode_id']=='a' else 1)+(.2 if feature=='control' else 0)
                        rr.append(dict(episode_id=e['episode_id'],episode_uid=e['episode_id'],scene_id='full/s',anchor=t,source=s,lag=t-s,
                            lag_group=group,lag_bucket=lag_bucket(t-s),feature=feature,layer=layer,fold=0,n_candidate_frames=e['gt_action_count'],**{k:value for k in report.METRICS}))
                with (folder/'pairs.csv').open('w') as f:
                    w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
                (folder/'COMPLETE.json').write_text(json.dumps(dict(folds=2,pairs=len(rr),pair_metrics_sha256=sha256(folder/'pairs.csv'))))
        return panel
    def test_real_schema_episode_aggregation_control_pairing_and_hash_rejection(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(report,'MODELS',('M5_E1',)):
            root=Path(tmp);panel=self.fixture(root)
            collector,contrasts,files=report.aggregate_probes(root/'probes',panel)
            summary=collector.summaries()
            r=report.unique(summary,model='M5_E1',feature='state',layer=24,lag='C',metric='cosine')
            self.assertAlmostEqual(r['estimate'],.5);self.assertEqual(r['n_episodes'],2)
            r=report.unique(summary,model='M5_E1',feature='memory_specific',layer=24,lag='C',metric='cosine')
            self.assertAlmostEqual(r['estimate'],-.2)
            files[0].write_text(files[0].read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'Hash mismatch'):report.aggregate_probes(root/'probes',panel)
    def test_r4_equal_layer_episode_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'episodes.csv';rr=[]
            for model in ('M5_E1','G3_M5'):
                for eid in ('a','b'):
                    for layer in range(3,32,4):
                        rr.append(dict(diagnostic='D-c',model=model,category='total_bank',role='combined',metric='mass',scope='query_head_mean',episode_id=eid,scene='s',layer=layer,value=.2 if model=='M5_E1' else .1))
            with p.open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
            result=report.pooled_bank_contrast(p);self.assertAlmostEqual(result['estimate'],-.1);self.assertEqual(result['n_episodes'],2)
    def test_incomplete_input_fails_before_creating_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pre=root/'prerequisites.json';pre.write_text('{"complete":false}')
            output=root/'new-stage'
            with patch('sys.argv',['report','--post-root',str(root),'--panel',str(root/'panel.json'),'--prerequisites',str(pre),'--output',str(output)]):
                with self.assertRaisesRegex(ValueError,'prerequisites incomplete'):report.main()
            self.assertFalse(output.exists())
if __name__=='__main__':unittest.main()
