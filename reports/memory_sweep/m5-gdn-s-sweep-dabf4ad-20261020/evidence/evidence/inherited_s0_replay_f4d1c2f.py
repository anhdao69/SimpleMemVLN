import json,sys
from qwen_vl.train.vln_runtime import load_checkpoint
from scripts.vln import check_episodic as replay
M5="/mnt/data/vmo-ai-task/anhdh35/m5-one-epoch-worker2-20261018/publication-m5-epoch1-20261019/epoch-1"
MODEL="/mnt/data/vmo-ai-task/anhdh35/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MAN="/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl"
import torch; torch.manual_seed(429)
model,serializer=load_checkpoint(M5,MODEL);model.eval()
cfg=json.loads(open(M5+"/navigation.json").read())["config"]
replay.cfg_for=lambda arm:cfg
for steps in (64,128):
    for dataset in ("r2rce","rxrce"):
        ep=replay.fixture(MAN,dataset,steps,rng_seed=steps)
        r=replay.gate_c(model,serializer,"m5_e1",ep,next(model.parameters()).device)
        print(json.dumps(dict(source="f4d1c2f",steps=steps,dataset=dataset,uid=ep["episode_uid"],**{k:r[k] for k in ("passed","rms_logit_error","max_logit_error","argmax_agreement","plan_id_mismatches","mean_selected_overlap","max_resident_kv")})),flush=True)
