# ActiveVLN Reference Analysis (GRPO stage)

Read-only analysis of the official ActiveVLN code, written to support a faithful re-implementation of its GRPO stage for SimpleMem (a Qwen3.5-4B streaming policy that emits one discrete action per step, with a KV window of 8 observation/action groups).

## 0. Provenance

| Item | Value |
|---|---|
| Repo | https://github.com/arvillion/ActiveVLN |
| Local clone | `/data/anh/code/ActiveVLN` (fresh clone; the path did not exist before) |
| Commit (`git rev-parse HEAD`) | `3a0c63b00e4f42c828cc74c3554afce17641da60` |
| Commit date | 2026-02-11T15:03:49+08:00 ("Add conference reference to project title in README"); 13 commits in total |
| Paper | arXiv:2509.12618 v1 (16 Sep 2025), "ActiveVLN: Towards Active Exploration via Multi-Turn RL in Vision-and-Language Navigation", ICRA 2026 |
| Dataset file hashes (sha256) | `data/r2r_4000_train.parquet` f83cd5f6210f1ecba84ddc9993de53662d0fbfc5feeba815363c40692868d94a; `data/r2r_val_tiny.parquet` c4d2490a46148d7801b123c4c007ce811a2c1a43f828412357702acd0a9e9bb0; `data/rxr_4000_train_new.parquet` 771121f25d4b497622f1e52d33452a7fe0c9c4a218aef94ada0b8774fffdbdd2 |

**Precedence rule used in this document.** Where the code and the paper disagree, the code wins, and the conflict is flagged. All paths below are relative to `/data/anh/code/ActiveVLN`. "Lxx" means a line number at the commit above.

---

## 1. Repo layout, dependencies, launch scripts, configs

### 1.1 Layout

| Path | Role |
|---|---|
| `verl/` | Vendored fork of verl, version string `0.4.1.dev` (`verl/version/version`). The upstream "Upload verl codebase" commit is be68a36. ActiveVLN changes it in the places listed below. |
| `verl/workers/agent/parallel_env_vlnce.py` | **Core multi-turn agent rollout loop**: `agent_rollout_loop`, `ParallelVlnceEnv`, dynamic sampling, and the (disabled) GT fallback. |
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | vLLM engine setup (L159-192). Calls `agent_rollout_loop` when `agent.activate_agent` is set (L307-316). |
| `verl/trainer/ppo/ray_trainer.py` | Training loop `fit()` (L1118-1625). Computes the per-episode step budget, i.e. the dynamic early-stopping budget (L1229-1308), pre-warms the next batch's scenes (L1349-1370), and contains `_validate` (L685-881). |
| `verl/trainer/ppo/core_algos.py` | GRPO advantage (L219-273), policy loss (L616-688), `agg_loss` (L580-613), KL estimators (L907-942). |
| `verl/workers/actor/dp_actor.py` | `update_policy` (L378-557) and `_optimizer_step` (L271-287). |
| `verl/workers/reward_manager/naive.py` | Adds `env_reward` into the token-level reward tensor (L61-64). |
| `verl/utils/reward_score/__init__.py` | Rule score for `data_source in ["r2r","rxr"]` is `0.0` (L118-119), so **all reward comes from the environment**. |
| `vlnce_server/` | HTTP environment server: Flask + Ray, run in a separate conda env with habitat 0.1.7. |
| `vlnce_server/env.py` | `VLNDataset`, `Simulator` (Ray actor that wraps `habitat.Env`), and `VLNCEEnv` (step, parse, reward). |
| `vlnce_server/service.py`, `server.py`, `client.py` | Actor pool (simulator reuse keyed by `(data_source, episode_id)`), batch endpoints, and the HTTP client. |
| `vlnce_server/prompt.py`, `utils/parse_utils.py`, `constants.py`, `env_config.py` | Prompt templates, action regex, reward-type constants, and per-env config dataclass. |
| `vlnce_server/VLN_CE/` | Vendored VLN-CE (habitat_extensions + vlnce_baselines configs). |
| `examples/vlnce/` | Launch scripts and the RL config. |
| `eval/vlnce/eval_vlnce.py`, `analyze_results.py` | Evaluation: an OpenAI client against `vllm serve`. |
| `data/` | RL training and tiny-validation parquet files. |

### 1.2 Dependencies

There are two separate environments (README "Quick Start").

**Environment server (`server`, Python 3.10)**
- habitat-sim **v0.1.7**, built from source with `--headless`.
- habitat-lab **v0.1.7**, built from source, including the `habitat_baselines` rl/ddppo requirements.
- `dtw`, `fastdtw`, `gym==0.17.3`.
- `numpy==1.23.0` and `numba==0.60.0`, pinned "to prevent a known segmentation fault".
- `pip install -e ./vlnce_server`. The server also needs flask, hydra, and ray (UNVERIFIED versions; not pinned anywhere).

**Trainer (`train`, Python 3.10)**
- `pip install --no-deps -e .` (the verl fork), then `scripts/install_activevln.sh`. That script pins:
  - `vllm==0.8.5.post1`, `torch==2.6.0`, `torchvision==0.21.0`, `tensordict==0.6.2`, `transformers[hf_xet]==4.51.3`, `numpy<2.0.0`
  - `flash_attn-2.7.4.post1` (cu12, torch2.6, cxx11abiFALSE wheel), `flashinfer 0.2.2.post1+cu124`
  - `qwen-vl-utils`, `ray[default]`, `swanlab`
- The logger is `['console','swanlab']`. `ray_trainer.py` L1589 imports `swanlab` unconditionally inside the agent-metrics block.

### 1.3 Launch scripts and configs

**RL training**
- R2R: `bash examples/vlnce/r2r.sh`. The README says `examples/vlnce/train_r2r.sh`, but **that file does not exist**.
- RxR: `examples/vlnce/rxr.sh`.
- Both run `python3 -m verl.trainer.main_ppo --config-path examples/vlnce --config-name train_vlnce_4gpus.yaml` with the overrides below.

`r2r.sh` L19-34 overrides:
```
data.train_files=[data/r2r_4000_train.parquet]  data.val_files=[data/r2r_val_tiny.parquet]
actor_rollout_ref.model.path=Arvil/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn
actor_rollout_ref.rollout.agent.base_url="http://127.0.0.1:5001"  ...agent.timeout=180
...agent.gen_length_tolerance=2.0  ...agent.enable_dynamic_sampling=true
...agent.reward.reward_type=weighted_success_ndtw  ...reward.success_reward_base=15  ...reward.ndtw_reward_base=0
```
- RxR uses `data/rxr_4000_train_new.parquet` (the README says `rxr_4000_train.parquet`) and `Arvil/Qwen2.5-VL-3B_sft_r2r_envdrop_rxr_multiturn`.

**RL config:** `examples/vlnce/train_vlnce_4gpus.yaml`. It inherits `verl/trainer/config/ppo_trainer.yaml`. Every value is listed in section 9.

**Environment server**
```
python3 -m vlnce_server.server server.host=0.0.0.0 server.port=5001 vlnce.gpus=[0,1] vlnce.r2r_gpu_plan=[16,16] vlnce.rxr_gpu_plan=[0,0]
```
- This runs 32 R2R simulators, 16 per GPU.
- The defaults live in `vlnce_server/config/server.yaml`.
- The server loads these habitat configs:
  - `vlnce_server/VLN_CE/vlnce_baselines/config/r2r_baselines/activevln_r2r.yaml`, which points to `habitat_extensions/config/vlnce_task_activevln_r2r.yaml` (**SPLIT: train**).
  - The RxR equivalents `activevln_rxr.yaml` → `vlnce_task_activevln_rxr.yaml`.

**Evaluation**
- `examples/vlnce/eval_r2r.sh` with `vlnce_baselines/config/r2r_baselines/activevln_r2r_test.yaml`, which points to `vlnce_task_activevln_r2r_test.yaml` (**SPLIT: val_unseen**).

**SFT / IL stage:** not released ("We do not provide the codebase for SFT"; README "For SFT Reference").
- Base model: `Qwen/Qwen2.5-VL-3B-Instruct`.
- R2R SFT data: R2R + R2R-EnvDrop. The README links both to the same Google Drive id (`1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr`).
- RxR SFT data adds RxR from HF `cywan/StreamVLN-Trajectory-Data/RxR`, converted to the R2R action space.
- Paper (Sec. V-C2):
  - The IL peak LR is 2e-5 with global batch 64.
  - The vision encoder is frozen; the connector and LLM are trained.
  - About 20 h on 4×L40S.
  - 167.6k trajectories (10.8k R2R, 146.3k EnvDrop, 10.5k RxR). Table II says 177K.
  - Epochs: not reported.
- The SFT prompt/target format must match the RL multi-turn format (README: "Ensure you use the appropriate system prompt").

---

## 2. Model, action space, prompt, context

### 2.1 Models and checkpoints

- **Base model:** Qwen2.5-VL-3B-Instruct.
- **RL init (R2R):** HF `Arvil/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn`. The README gives SR ~38.5.
- **Released RL checkpoint (R2R):** `Arvil/Qwen2.5-VL-3B_rl_r2r_4000`, at **step 350**.
- **RxR:** init `Arvil/Qwen2.5-VL-3B_sft_r2r_envdrop_rxr_multiturn` (SR ~41.0); RL checkpoint `Arvil/Qwen2.5-VL-3B_rl_rxr_4000_step350`.
- **HF collection:** https://huggingface.co/collections/Arvil/activevln-68e7367d8bd2b426985f0c4a

### 2.2 Action space and output format

The policy emits **macro actions as plain text**, with **up to 3 macro actions per turn**, separated by `,`.

R2R regex (`vlnce_server/utils/parse_utils.py` L9-10):
```
^((move forward) (25|50|75)cm|((turn left)|(turn right)) (15|30|45) degrees|(stop))$
```
- This gives 10 tokens of vocabulary: `move forward 25cm/50cm/75cm`, `turn left 15/30/45 degrees`, `turn right 15/30/45 degrees`, `stop`.
- The RxR action space uses 30/60/90 degrees, but `ParallelVlnceEnv.reset` hard-codes `action_space="r2r"` (`parallel_env_vlnce.py` L906), even for RxR training. The RxR parquet also has an `action_space` column equal to `r2r`.

Each macro action is executed as repeated simulator primitives (`env.py` L593-612):
- `move forward X` → `X//25` × MOVE_FORWARD (0.25 m each).
- `turn Y` → `Y//15` × TURN (15° each).
- `stop` is **not executed in the simulator** during training (L426-428). Success is judged from the current `distance_to_goal`.
- Max per turn: 3 macros = up to 9 primitives.

Configuration: `max_actions_per_step: 3` and `action_sep: ","` (yaml L87-88). An example response looks like `move forward 75cm, turn left 30 degrees, move forward 50cm<|im_end|>`. No `<think>` and no tags (`prompt_format: no_think_no_tag`).

### 2.3 Prompt format (training)

**System message.** It is the only message in the parquet `prompt` column (verified, identical across all 4000 rows):
```
You are a helpful assistant. Your goal is to follow the given instruction to reach a specified destination. \nAt each step, you receive a first-person image (starting view if first step (step 1), or post-action view otherwise). Your task is to select choose one action from: move forward 25cm, move forward 50cm, move forward 75cm, turn left 15 degrees, turn left 30 degrees, turn left 45 degrees, turn right 15 degrees, turn right 30 degrees, turn right 45 degrees, or stop. \nThe instruction will be provided with each observation. You can take multiple actions at each turn. 
```
The modified `rl_dataset.py` (L216) sets `add_generation_prompt = (last role == "assistant")`, which is False here. The dataset prompt is therefore just `<|im_start|>system\n...<|im_end|>\n`, left-padded to `max_prompt_length=1024`.

**First user turn** (`vlnce_server/prompt.py` L18-22 plus `format_prompt`; `env.py` L678-687):
```
[Initial Observation]:\n<image>\nInstruction: {instruction}\nDecide your next action. \nYou can take up to 3 actions at a time, separated by ','. 
```

**Later user turns:**
```
After that, the observation is:\n<image>\nInstruction: {instruction}\nDecide your next action. \nYou can take up to 3 actions at a time, separated by ','. 
```
- The instruction is repeated every turn.
- The observation tokens are produced by `tokenizer.apply_chat_template([{"role":"user",...}], add_generation_prompt=True)`. The auto-inserted default system block is then removed by `_strip_system_block` (`parallel_env_vlnce.py` L20-30, L843-844).
- Each observation chunk is therefore `\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`.

### 2.4 Images

- **Raw frame:** 640×480 RGB (`env.py` L672 drops the alpha channel).
- **Serialization:** PNG between server and trainer.
- **Resizing:** `qwen_vl_utils.fetch_image` with `max_pixels=76800`, `min_pixels=1024` (yaml L75-76; `parallel_env_vlnce.py` L83-110). By qwen-vl-utils `smart_resize` arithmetic, 640×480 becomes **308×224**, i.e. 11×8 = **88 visual tokens per frame**. This was derived by hand, not run.
- vLLM uses the same `mm_processor_kwargs` (`vllm_rollout_spmd.py` L182-191) and `limit_mm_per_prompt image=200`.

### 2.5 Context management

- **Full history, no windowing.** Every turn appends observation tokens (one new image) after the previous response. All images accumulate in `multi_modal_data` (vLLM) and `multi_modal_inputs` (trainer) (`parallel_env_vlnce.py` L291-323).
- **Turn and length limits:**
  - `max_turn_budget: 40` (yaml L82). The loop runs `range(max_turn_budget+1)` (L235), and turn 41 is force-terminated by the env (`env.py` L408).
  - Per-turn generation is capped at `single_response_max_tokens: 512` (yaml L67, used L122).
  - The total sequence cap is `max_prompt_length + max_response_length = 1024 + 25480`. Exceeding it hits `assert False`; nothing is truncated.
  - vLLM `max_model_len` is hard-coded to 32768 (`vllm_rollout_spmd.py` L170).
- **Optional GT prefix ("history_actions"):** the env can replay a ground-truth action prefix before the policy starts (`env.py` L255-315). With the shipped config this is **disabled**: `warmup_ratio=0.0` makes `n_drop = n_actions`, so `history_actions=[]` (see §8). Every RL rollout starts from the episode's start pose.

---

## 3. Environment

### 3.1 Habitat task config

The train config is `vlnce_server/VLN_CE/habitat_extensions/config/vlnce_task_activevln_r2r.yaml`. The test config `..._r2r_test.yaml` is identical except SPLIT is val_unseen and `TOP_DOWN_MAP_VLNCE` is enabled.

| Setting | Value (line) |
|---|---|
| habitat | 0.1.7, `ACTION_SPACE_CONFIG: v0`, `ALLOW_SLIDING: True` (L4, L11) |
| FORWARD_STEP_SIZE | 0.25 m (L7) |
| TURN_ANGLE | 15° (L8). The RxR config also uses 15°; RxR macros 30/60/90 are 2/4/6 primitives. |
| RGB sensor | 640×480, **HFOV 90**, position [0, 1.25, 0] (L12-17). No depth. |
| SUCCESS_DISTANCE | 3.0 m (TASK, SUCCESS, SPL) |
| MAX_EPISODE_STEPS | 500 primitives (L2) |
| Measures | DISTANCE_TO_GOAL (habitat geodesic), SUCCESS, SPL, NDTW (fastdtw, GT from `{split}_gt.json.gz`), PATH_LENGTH, ORACLE_SUCCESS (3.0 m default), STEPS_TAKEN |
| Dataset | `VLN-CE-v1`, `data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz`, scenes `data/scene_datasets` (MP3D) |

### 3.2 Server architecture (`vlnce_server/server.py`, `service.py`, `env.py`)

- **Startup.** `ray.init` with custom resources `r2r_gpu{i}` and `rxr_gpu{i}`. One `VLNDataset` Ray actor per dataset (`env.py` L34-74) loads all episodes plus the GT locations.
- **Simulator pool.** `ActorPool.init_pool` creates `Simulator` Ray actors, each holding one `habitat.Env`, according to the per-GPU plan (`service.py` L27-38).
- **Scene caching.** Simulators are tracked in `free_by_key[cache_key]`, where `cache_key = sha256(data_source, episode_id)` (`env_config.py` L40-44). `acquire_many` prefers a simulator that last loaded the same episode and otherwise falls back to any free one (L40-82).
- **Scene pre-loading.** At each step, the trainer sends the *next* batch's episodes to `/batch/warmup` (`ray_trainer.py` L1349-1370; `service.py` L230-258). Each listed episode gets `start_new_episode` called on a free simulator.
- **Per-env objects.** A `VLNCEActor` wraps `VLNCEEnv(cfg, sim)`. At construction it starts the episode, replays history actions, and saves a state checkpoint (`save_current_state`). `reset()` restores that checkpoint (`env.py` L318-333; `load_state` L181-203), so the n=4 group replicas share one identical start.
- **HTTP endpoints.** `/environments`, `/batch/reset`, `/batch/step`, `/batch/close`, `/batch/warmup`. Note that `start(background=True)` polls `/health`, but that route is commented out; the shipped launch uses `background=False`.
- **Trainer side.** `ParallelVlnceEnv` (`parallel_env_vlnce.py` L764-1023) reuses envs whose `config_id` (episode_id, history_actions, data_source) matches, and closes the others. Only TP rank 0 steps the env; results are broadcast (L247-252).

### 3.3 Action parsing and invalid output (`env.py` L335-456; `parse_utils.py` L30-65)

1. The trailing `<|im_end|>` is stripped (`parallel_env_vlnce.py` L814-818).
2. The response is split on `,` and each piece is stripped. `format_correct = len(actions) <= 3`. If there are more than 3 actions, the list is truncated, **but `format_correct` stays False**.
3. `format_correct` also requires at least one action and every action matching the regex. Otherwise it is False.
4. Termination order (L408-456):
   - If `turn > turn_budget` → **done, fail**, reason "number of turns exceeded".
   - Else if not `format_correct` → **done, fail**, reason "unexpected format.". Nothing is executed and the reward is 0, so **any malformed output kills the episode**.
   - Else execute macros in order. For each macro: `local_step += 1`. If the macro is STOP, success is `distance_to_goal <= 3.0` (`_get_metrics` L614-619). Else if `local_step > step_budget` → done, fail, reason "number of steps exceeded". Remaining macros in the turn are dropped once done.
5. `format_reward = 0` (L911). There is no per-step shaping.
6. Habitat's own 500-primitive limit is never reached in practice, because step_budget ≤ 120 macros × 3 = 360 primitives. Note also that `env._episode_over` is not checked by the code.

---

## 4. Datasets

### 4.1 RL training set: `data/r2r_4000_train.parquet` (in the repo)

There is no generation script in the repo. The paper says only "we sample 4k navigation instructions from R2R and RxR, respectively". The selection procedure is UNVERIFIED.

What follows was verified by decoding the parquet with a pure-Python reader (nothing installed) and cross-checking against `/data/anh/datasets/simplemem/datasets/R2R_VLNCE_v1-3_preprocessed`.

**Schema:** `data_source`='r2r', `prompt` (one system message), `ability`='vln', `env_name`='vlnce', `reward_model{ground_truth:'', style:'model'}`, `extra_info{episode_id:int, gt_actions:list[str], split:'train'}`.

**Episodes**
- 4000 rows with **4000 unique `episode_id`s, all present in R2R-VLNCE v1-3 `train`** (10,819 episodes).
- They cover **all 61 train scenes**, with 2706 distinct `trajectory_id`s.
- Mean geodesic distance is 9.257 m (full train: 9.274 m). This is consistent with a uniform random sample, but the sampling method is UNVERIFIED.
- The rows are not sorted by id. The first ids are 5280, 455, 4809, 3000, 6195, ...

**Separation from validation**
- No scene overlap with val_unseen.
- No `trajectory_id` overlap with val_seen or val_unseen.
- One instruction string also occurs in val_unseen (a generic text).
- Numeric `episode_id` collisions with val ids are meaningless: ids are per-split, and the server loads only `SPLIT: train`.

**`gt_actions`**
- These are expert macro actions in the same 10-action text vocabulary, **ending with `stop`** (4000/4000).
- Length: min 10, max 115, mean 36.91, median 35 (p10 23, p90 53).
- Expanding the macros to primitives reproduces the non-STOP count of `train_gt.json.gz['actions']` **exactly for all 4000 episodes**.
- How primitives were grouped into 25/50/75 and 15/30/45 chunks is UNVERIFIED. Adjacent same-type macros are frequent (56,746 of 143,640 adjacent pairs), so it is not greedy maximal merging.
- `gt_actions` are used **only for `len()`**, to set the step budget, and for the disabled prefix/fallback features. The paper states that GT actions are not used in RL.

**Other files**
- `data/r2r_val_tiny.parquet`: 200 rows with `split:'val'`. The ids are all in the train split and disjoint from the 4K set. Because the server serves only the train split, this "val" set is a held-out subset of **train**. Intent is UNVERIFIED. It is only used by `_validate`, and validation is effectively disabled (`test_freq: 10000000`, `val_before_train: false`).
- `data/rxr_4000_train_new.parquet`: 4000 rows, `data_source`='rxr', extra `extra_info.difficulty` and `action_space` columns.

**Simulator episode data (server side):** R2R_VLNCE_v1-3_preprocessed from Google Drive `1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr`; RxR_VLNCE_v0 from `145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t`; MP3D scenes. There are no HF dataset ids for the RL data; it ships in-repo.

### 4.2 Evaluation splits

- R2R-CE **val_unseen**: 1839 episodes in the local copy (`activevln_r2r_test.yaml`).
- RxR-CE val_unseen: guide role, en-US/en-IN (`vlnce_task_activevln_rxr_test.yaml`).

---

## 5. Rollout generation

| Setting | Value | Source |
|---|---|---|
| Prompts per batch | `data.train_batch_size: 8` | yaml L13 |
| Group size n | `rollout.n: 4` → 32 trajectories/step | yaml L55 |
| Temperature | **1.2** | yaml L56 |
| top_p / top_k | 1.0 / -1 (ppo_trainer defaults; not overridden) | `ppo_trainer.yaml` |
| Validation sampling | temperature 0.2, top_p 0.95 (unused) | yaml L62-64 |
| Max tokens per turn | 512 | yaml L67 |
| Max turns | 40 (+1 forced-termination turn) | yaml L82 |
| Max macro steps | `min(2.0·len(gt_actions), 120)` per episode | §8 |
| vLLM | TP=1, `gpu_memory_utilization 0.6`, prefix caching on, chunked prefill off, `enforce_eager false`, `max_num_batched_tokens 32768`, `free_cache_engine true` (sleep mode), seed 0 | yaml L50-61; `vllm_rollout_spmd.py` L159-192 |

### 5.1 Multi-turn loop (`parallel_env_vlnce.py` L113-327)

1. `env.reset(prompts, n=4)` creates 4 independent envs per prompt, in interleaved order: `[p0×4, p1×4, ...]`.
2. Each trajectory starts as `running_states = dataset prompt ids` (left-padded system block), followed by the reset observation tokens.
3. On each turn, every active trajectory calls `vllm_engine.generate` with `n=1` (the group comes from the 4 replicas), `detokenize=True`, `skip_special_tokens=False`, and `include_stop_str_in_output=True`. The response therefore ends with `<|im_end|>`.
4. `env.step` returns `(obs, reward, done, info)`. The response tokens are appended with **action_mask=1**; the reward is added at the **last response token** (L266-272).
5. If not done, the next observation tokens (model tokenization, with expanded `<|image_pad|>`×88 per image) are appended with **action_mask=0**, plus the image and its `multi_modal_inputs` (L291-323). vLLM gets a parallel token list in which `<|vision_start|><|image_pad|><|vision_end|>` appears once per image, so vLLM expands it.
6. Episodes leave the active set when `done`. The loop ends when none are active or 41 turns have run.

### 5.2 Training sequence assembly (L709-761)

- Each sequence is `[prompt (1024, left-padded)] + [obs0, resp0, obs1, resp1, ..., resp_last]`, right-padded to `prompt_len + response_len`.
- `response` is the last 25480 tokens. It contains **all observation tokens and all policy tokens** of the episode.
- `action_mask` is 1 on policy tokens (including `<|im_end|>`) and 0 on observation, user-template and image tokens. It becomes `response_mask` (`ray_trainer.py` L203-218; `dp_actor.py` L468-472), so **loss, entropy and advantage apply only to policy tokens**.
- `attention_mask` is 1 everywhere real.
- Position ids use Qwen2-VL M-RoPE `get_rope_index` over the full multi-image sequence (L719-731).
- `env_reward` is a token tensor with the episode's scalar reward on the final response token. Non-terminal turns contribute 0 because `format_reward=0`.
- `task_success`, `episode_len`, `tool_cnt`, and the `info` dicts are carried along for logging.

### 5.3 Log-probs

- Old log-probs are recomputed by the FSDP actor (`ray_trainer.py` L1402-1412), not taken from vLLM (`calculate_log_probs: False`).
- Logits are divided by the rollout temperature 1.2 in both `compute_log_prob` and `update_policy` (`fsdp_workers.py` L783, L826; `dp_actor.py` L184, L263).

---

## 6. Reward function

The reward is computed **once, at episode termination**, inside the server (`vlnce_server/env.py` L367-406, `set_ending_signals`).

Shipped config: `reward_type=weighted_success_ndtw`, `success_reward_base=15`, `ndtw_reward_base=0`, `format_reward=0`.

```python
# env.py L376-400
d2g = metr["distance_to_goal"]          # habitat DistanceToGoal (geodesic, metres) at termination
...
elif self.config.reward_type == REWARD_WEIGHTED_SUCCESS_NDTW:
    reward_components["success_reward"] = is_goal_reached * self.config.success_reward_base * (1 - min(d2g / 3.0, 1))
    if reason in [REASON_SUCCESS, REASON_GOAL_NOT_REACHED]:
        reward_components["ndtw_reward"] = ndtw * self.config.ndtw_reward_base
    else:
        reward_components["ndtw_reward"] = 0.0
...
reward += sum(reward_components.values())
```
`is_goal_reached` is True only on an emitted `stop` with `distance_to_goal <= 3.0` (L431-446, L617).

**Effective reward:**
```
R = 15 · (1 − d_goal/3)   if the agent emits STOP with d_goal ≤ 3.0 m
R = 0                     otherwise (stop outside 3 m, format error, step budget exceeded, turn budget exceeded)
```
- R ranges from 15 down to 0 and is linear in the final geodesic distance.
- The nDTW term is multiplied by `ndtw_reward_base=0`, so it is zero.
- The nDTW variant would use `gen_ndtw`, which is nDTW of the generated path against the remaining GT path with a 3.0 normalizer (L625-644).
- There are no SPL, path-length, or format terms. Format errors are penalized only through early termination with R=0.

**Paper discrepancy.** Paper Eq. 8 is rendered as `R = 15·𝕀(success)·d_goal/3`. Taken literally, that would *increase* with distance. The code uses `(1 − min(d/3,1))`. **Follow the code.**

**Other reward types available** (`constants.py`):
- `success`: 15·𝕀.
- `weighted_success`.
- `success_ndtw`.
- The paper's ablation (Table III) shows soft > hard > path-alignment.

**Where the reward reaches the loss.** The reward sits on `env_reward[last policy token]`. `NaiveRewardManager` adds `env_reward` to a zero tensor (`naive.py` L58-64). The rule score for r2r is 0.0 and is added at `valid_response_length-1`. GRPO then sums over tokens (`core_algos.py` L247), giving one scalar per trajectory.

---

## 7. GRPO loss

| Item | Value | Source |
|---|---|---|
| Advantage estimator | `grpo` | yaml L23 |
| Group stats | Per `uid` (one uid per prompt, repeated n=4): `A = (R − mean_g) / (std_g + 1e-6)`. **`torch.std` is unbiased (n−1)**. If the group size is 1, mean=0 and std=1. | `core_algos.py` L219-273 |
| `norm_adv_by_std_in_grpo` | True (default) | `ppo_trainer.yaml` |
| Advantage broadcast | The scalar A is multiplied by `response_mask` (policy tokens only); every token of every turn gets the same A | L271 |
| Zero-variance groups | A=0, since (x−x)/(0+1e-6). They are **not filtered**; their tokens stay in the token-mean denominator | — |
| Policy loss | PPO clip with dual-clip: `clip_ratio_low 0.2`, **`clip_ratio_high 0.28`** (clip-higher), `clip_ratio_c 3.0`, log-ratio clamped to ±20 | yaml L40-41; `core_algos.py` L616-688 |
| Loss aggregation | `token-mean` (default), **computed per micro-batch** (2 sequences per GPU), then divided by gradient_accumulation=4 | `dp_actor.py` L499-541; `ppo_trainer.yaml` |
| KL | `use_kl_loss: false`, `kl_loss_coef 0.0` (type `low_var_kl`, unused); `algorithm.use_kl_in_reward` false; `kl_ctrl.kl_coef 0.0` | yaml L22-25, L37-39 |
| Reference model | **Not instantiated**: RefPolicy is created only if `use_kl_in_reward or use_kl_loss` | `main_ppo.py` L175-177 |
| Entropy coefficient | 0.0 (entropy is only logged) | yaml L42 |
| PPO epochs | 1 (default) | `ppo_trainer.yaml` |
| Mini-batch | `ppo_mini_batch_size: 8` prompts → ×n=4 = 32 trajectories, then ÷4 GPUs = 8 per GPU | yaml L35; `fsdp_workers.py` L172-173 |
| Micro-batch | `ppo_micro_batch_size_per_gpu: 2` → 4 accumulation steps | yaml L36 |
| Log-prob micro-batch | 8 per GPU (rollout and ref) | yaml L51, L101 |
| `balance_batch` | True (default): reorders trajectories across DP ranks by sequence length | `ray_trainer.py` L1385-1386 |
| Dynamic batch size | `use_dynamic_bsz: false` | default |

**Important consequence.** The mini-batch (32 trajectories) equals the full rollout batch, and there is 1 PPO epoch. So there is **exactly one optimizer step per training step, and it is fully on-policy**: `old_log_probs` are recomputed by the same weights just before the update. The ratio is therefore ≈1, apart from numerical noise between the two forward passes, and the clip (0.2/0.28) is effectively inactive.

The update is in practice a REINFORCE-style group-normalized policy gradient with micro-batch token-mean weighting. The paper's "clipping parameter 0.28" matches `clip_ratio_high`.

---

## 8. "Dynamic early stopping" and "dynamic sampling"

The paper names two distinct mechanisms. Both are implemented.

### 8.1 Dynamic early stopping: a per-episode step budget relative to expert length

Paper (Sec. IV-B3): `T_max = α_roll · |τ*|`, with α_roll = 2. A trajectory that exceeds it is stopped and marked failed (R=0). The paper says about 10% of rollouts are terminated this way, and Table V reports 10.9% faster rollouts.

Code (`ray_trainer.py` L1229-1297; config `gen_length_tolerance=2.0`, `smooth_alpha=2.0`, `min_gen_actions=2`, `max_step_budget=120`, `prob_from_scrath=0.0`, `warmup_ratio=0.0`, `random_drop=false`):
```python
warmup_steps = int(total_steps * warmup_ratio)            # = 0
if step > warmup_steps or random.random() < prob_from_scrath:   # always true (global_steps >= 1)
    n_drop = n_actions                                    # policy generates the whole episode
    n_limit = int(min(n_drop * tol, n_max))               # = int(min(2.0 * len(gt_actions), 120))
...
step_budget = n_limit
num_gt_actions = n_actions - n_drop                       # = 0  -> history_actions = []
turn_budget = max_turn_budget - ceil(0 / 3) = 40
```

- **|τ\*|** is the number of **macro** GT actions in the parquet, **including the final `stop`**. For the 4K set the median is 35, so the budget is 70 macros.
- 163 of 4000 episodes have `2·len > 120` and are capped at 120.
- The budget counts **macro actions** (each comma-separated action, STOP included), not simulator primitives.
- Enforcement (`env.py` L447-452): after a non-STOP macro has executed, if `local_step > step_budget` the episode ends with reason "number of steps exceeded" and reward 0.
- Separately, the turn budget (40) ends episodes at turn 41.

**Dormant curriculum.** With `warmup_ratio > 0`, the trainer would replay a GT prefix of `n_actions − n_drop` actions, where `n_drop` grows as `n_min + (n_actions − n_min)·(log step / log(T_w+1))^α`. It is disabled in the shipped config.

**Validation budgets** are fixed at `max_step_budget=120` and `max_turn_budget=40` (L744-745).

### 8.2 Dynamic sampling: re-roll groups with zero successes

Code: `parallel_env_vlnce.py` L330-532; config `enable_dynamic_sampling=true` (r2r.sh), `max_sample_attempts: 2`.

```python
group_success_count = task_success.reshape(B, n).sum(1)
groups_need_resample = where(group_success_count < 1)   # all 4 failed
# re-reset envs and re-run the full multi-turn rollout for those groups (all n=4), then
running_states[group slice] = ds_running_states[...]    # REPLACE the whole group, even if the new group also failed
```

- Up to 2 extra attempts; the loop stops early once no group has zero successes.
- It is **not** DAPO-style filtering: the batch size stays 8×4, groups are never dropped, and all-success groups are kept.
- A group that is still zero-success after 2 attempts stays in the batch with A=0.
- Training only; it is skipped when `meta_info.validate` is set.
- There is a logging-key mismatch: the code sets `from_dynamic_sampling`, while the logger reads `is_dynamic_sampling`. This affects captions only.
- The paper's Table III ablations were run *without* dynamic sampling (SR 48.1 vs 50.1 with it).

### 8.3 GT fallback (disabled)

`enable_fallback: false` (yaml L79; `parallel_env_vlnce.py` L534-707). When enabled, it would replace the last trajectory of each zero-success group with a scripted GT trajectory.

---

## 9. Optimizer and all hyperparameters (R2R RL run)

| Group | Setting | Value |
|---|---|---|
| Optimizer | AdamW | lr **1e-6**, betas (0.9, 0.999), weight_decay **0.01** (`fsdp_workers.py` L418-423; ppo_trainer defaults) |
| LR schedule | constant | `warmup_style constant`, `lr_warmup_steps -1`, `lr_warmup_steps_ratio 0.0` → **no warmup, constant LR** |
| Gradient clipping | `grad_clip` | **1.0**. A non-finite grad norm skips the step (`dp_actor.py` L271-287). |
| Data | batch / shuffle | `train_batch_size 8`, `shuffle: false` (SequentialSampler, `main_ppo.py` L273-283), `drop_last=True` (`ray_trainer.py` L590) |
| Steps | length | `total_epochs: 1` → 4000/8 = **500 steps**. The released checkpoint is **step 350**; the paper says the reward converges around step 300 and SR peaks around step 350. |
| Rollout | group / budget | n=4, temperature 1.2, top_p 1.0, 40 turns, 3 macros/turn, 512 tokens/turn |
| Sequence | lengths | `max_prompt_length 1024`, `max_response_length 25480`, vLLM `max_model_len 32768`, `truncation: error`, `filter_overlong_prompts: true` |
| Checkpointing | save | `save_freq 50`; `save_contents ['model','optimizer','extra','hf_model']` |
| Validation | test | `test_freq 10000000`, `val_before_train false`, so validation is effectively off |
| Hardware | trainer | `n_gpus_per_node 4`, `nnodes 1`: 4×L40S 48 GB with 512 GB CPU RAM (README). The env server runs 32 simulators on 2 GPUs. RL takes about 20 h (paper). |
| FSDP | strategy | **fsdp2** for actor, ref, critic and RM. `param_offload: true`, `optimizer_offload: true`; ref `param_offload: true` (ref unused). `reshard_after_forward` True. `ulysses_sequence_parallel_size 1`. |
| Precision | dtypes | Actor master weights are fp32 (`fsdp_workers.py` L243-245). Mixed precision is bf16 params, fp32 reduce, fp32 buffers (L337-347). vLLM runs in bf16. |
| Model flags | | `use_remove_padding true`, `use_fused_kernels true`, `use_liger false`, `enable_gradient_checkpointing true` (default) |
| Freezing | vision tower | **Not frozen in RL.** The AdamW optimizer covers all `actor_module_fsdp.parameters()`, and no requires_grad logic exists on the FSDP path. The paper says "all parameters trainable" in RL; in IL the vision encoder is frozen. |
| Critic | | none (GRPO); `critic_warmup 0` |

---

## 10. Evaluation

### 10.1 Protocol (`eval/vlnce/eval_vlnce.py`, `examples/vlnce/eval_r2r.sh`)

**Model server.** The model is served with:
```
vllm serve $MODEL --limit-mm-per-prompt image=200,video=0 --mm_processor_kwargs '{"max_pixels": 80000}' --max-model-len 32768 --enable-prefix-caching
```
The client pre-resizes frames with `smart_resize(max_pixels=76800)`, giving 308×224 (L315-321).

**Episodes.** All R2R val_unseen episodes, split across `--split-num 32` processes (`get_splits(allow_uneven_splits=True)`), running `habitat.Env` locally (habitat 0.1.7, `activevln_r2r_test.yaml`).

**Decoding.** **Sampled, not greedy**: temperature 0.2, top_p 0.8, max_tokens 512, n=1 (L167-172). There is a single generation per episode (`--num-generations 1`), and no seed is passed to the API.

**Context.** Full chat history with every image (OpenAI messages, chat template applied by vLLM).

**Eval prompt text differs from training.** The user content is three items:
- text `"[Initial Observation]:"` (or `"After that, the observation is:"`)
- the image
- text `"Instruction: {instr}Decide your next action. You can take up to 3 actions at a time, separated by ','. "`

Unlike training, there are no `\n` separators and **no space between the instruction and "Decide"** (L45-51, L344-348).

**Parsing is lenient** (L200-233, L371-393):
- The output is split on `", "`.
- Any substring "stop" → STOP. "forward"/"left"/"right" plus the first integer → `min(3, int(v/25 or v/15))` primitives.
- Unparseable output → **one random primitive from {forward, left, right}**.
- Training, by contrast, terminates the episode on bad format.

**Limits**
- `--max-turns 120` (model calls; forced STOP after that).
- `EARLY_STOP_STEPS 400` primitives → forced STOP.
- `EARLY_STOP_ROTATION 25`: more than 25 consecutive primitives with unchanged `distance_to_goal` → forced STOP.
- habitat `MAX_EPISODE_STEPS 500`.

**Metrics (`analyze_results.py`)**
- Reported: `success` (SR), `spl`, `distance_to_goal` (NE), `oracle_success` (OSR), `path_length`, as means over episodes.
- **nDTW is not saved or reported by this script.**
- `inf`/`nan` values are mapped to 0 via `check_inf_nan`. This would bias NE downward if d2g were ever inf.
- Per-episode logs go to `log/stats_{id}_{gen}.json`, with assistant turns in `extra_info/`.

### 10.2 Published numbers (paper v1; R2R-CE val-unseen, Table I; Qwen2.5-VL-3B)

| ActiveVLN stage | NE↓ | OSR↑ | SR↑ | SPL↑ |
|---|---|---|---|---|
| IL (SFT checkpoint `..._sft_r2r_envdrop_multiturn`; README "SR ~38.5") | 6.83 | 42.7 | 38.5 | 33.7 |
| IL + DAgger | 6.11 | 50.2 | 45.4 | 41.5 |
| **IL + RL (GRPO, 4K R2R; release `..._rl_r2r_4000`, step 350)** | **5.31** | **58.3** | **50.1** | **43.7** |

nDTW is not reported for R2R.

**Ablations (R2R val-unseen)**
- Table III, without dynamic sampling:
  - soft success reward: 5.56 / 56.4 / 48.1 / 40.6
  - hard success reward: 46.2 SR
  - path-alignment variants: 44.4–46.3 SR
- Table IV, single-turn paradigm: IL 39.7 SR → IL+RL 40.3 SR. The multi-turn paradigm goes 38.5 → 50.1.
- Table V, average rollout time: 106.9 s. It rises to 118.6 s without dynamic early stopping, 111.9 s without scene pre-loading, and 109.3 s without scene caching.

**RxR-CE val-unseen (Table II; 3B; 177K IL data)**
- w/o RL: NE 7.25, SR 41.0, SPL 34.1, nDTW not reported.
- IL+RL: NE 5.84, SR 50.7, SPL 41.2, nDTW 58.1.
- The RxR RL init is `..._sft_r2r_envdrop_rxr_multiturn`.

Paper numbers were taken from the arXiv HTML (v1) via a summarizing fetch. Re-check against the PDF before citing in a publication; marked partially UNVERIFIED for exact digits.

---

## 11. Other reproduction notes and caveats

1. **Seeds and determinism.**
   - vLLM seed is 0 (`config.get("seed",0)`); the dataloader generator seed of 1 is unused because shuffle is false.
   - No global torch/numpy seed is set by the VLN path, and the env reset seed of 42 is ignored.
   - Rollouts are therefore not bit-reproducible. Exact determinism is UNVERIFIED.
2. **README errors.**
   - `train_r2r.sh` is actually `examples/vlnce/r2r.sh`.
   - `rxr_4000_train.parquet` is actually `rxr_4000_train_new.parquet`.
   - The R2R-EnvDrop link equals the R2R link.
3. **Hard-coded Qwen2.5 token ids** in `ParallelVlnceEnv.reset`: 151644 `<|im_start|>`, 77091 `assistant`, 198 `\n`, 151645 `<|im_end|>` (L990-993). They are used to build the GT-prefix response mask. Any new tokenizer must update them, or the prefix logic must be removed.
4. **`_strip_system_block`** assumes the chat template injects a default system block for user-only messages (true for Qwen2.5-VL). Check this for Qwen3.5.
5. **Length overflow** raises `assert False` instead of truncating (`parallel_env_vlnce.py` L198-326). With full-history context, 41 turns × (88 image tokens + ~60 text tokens + response) fits comfortably in 26.5k tokens. That figure is estimated, UNVERIFIED.
6. **`_validate` is stale.** It reads `info['success']` and `info['env_step']`, but env info provides `task_success` and `env_local_step`. Running validation would likely `KeyError` (UNVERIFIED; it is never triggered with the shipped config).
7. **ndtw monkey-patch.** The `NDTW` measure is re-registered per simulator to share GT paths through the Ray dataset actor (`env.py` L94-142). `VLNDataset` keeps only 2 episodes in the habitat dataset object to save memory; episodes are fetched by id.
8. **Simulator state restore** (`load_state`) restores agent pose and selected metrics without re-reconfiguring the scene. Group replicas therefore start from identical states.
9. **RxR training uses the R2R action space** (hard-coded `action_space="r2r"`).
10. **The metrics module** (`metric_utils.compute_agent_metrics`) and swanlab logging are required imports. A rollout dump is appended to `{default_local_dir}/rollout.jsonl` every step.

---

## 12. Implications for porting to a one-action-per-step streaming policy (SimpleMem, Qwen3.5-4B, KV window = 8 groups)

### Can be kept identical

- **GRPO math**:
  - Group size n=4 over 8 prompts/step (32 trajectories).
  - Unbiased group std with ε=1e-6 and `norm_adv_by_std=True`.
  - Episode-level scalar advantage broadcast to every policy (action) token of every step.
  - Clip 0.2/0.28 with dual-clip c=3.
  - No KL, no reference model, entropy coefficient 0.
  - 1 PPO epoch, 1 optimizer step per batch (on-policy).
  - Token-mean aggregation. To match exactly, replicate the per-micro-batch token-mean with 2 trajectories/GPU and accumulation 4, or document the change.
- **Optimizer**: AdamW, LR 1e-6 constant, no warmup, weight decay 0.01, grad-clip 1.0, skip on non-finite grad norm. 500 steps over the 4K set in fixed order, with checkpoints every 50 and step 350 as the reference point.
- **Sampling**: temperature 1.2, top_p 1.0, and old/new log-probs computed at the same temperature (logits/1.2).
- **Reward**: R = 15·(1 − d_goal/3) on STOP within 3 m, else 0, computed at termination from habitat's geodesic `distance_to_goal`. No format or shaping terms.
- **Dynamic sampling**: up to 2 re-rolls of groups with zero successes, replacing the whole group, with no group filtering.
- **Training data**: the same 4000 R2R train episode ids (`data/r2r_4000_train.parquet`, sha256 f83cd5f6...). Instruction text comes from the simulator episode, not the parquet.
- **Masking principle**: loss only on policy-generated action tokens (including the end-of-turn token); zero on observation, image and template tokens.

### Must change or be decided explicitly

1. **Step and budget units.**
   - ActiveVLN steps are *macro* actions (up to 75 cm or 45°), up to 3 per model call.
   - SimpleMem emits one primitive per call (25 cm / 15°).
   - The faithful translation of T_max = 2·|τ*| is **2 × (number of GT primitives + 1 for STOP)**. That is 2 × (`len(train_gt.json.gz[id]['actions'])`, which includes the stop) in primitive units, which is about 2–3× larger than the macro count.
   - ActiveVLN's cap of 120 macros corresponds to roughly 360 primitives of motion. Decide whether to cap at 500 (habitat MAX_EPISODE_STEPS) or at a scaled cap; document the choice.
   - The macro-unit budget (`int(min(2·len(gt_actions),120))` on macro counts) can also be reproduced literally, but that gives a much tighter primitive budget. **Recommendation:** use primitive-unit α=2 and record the deviation.
2. **Turns.** The 40-turn / 120-step budgets become a single primitive-step budget; there is no separate turn budget. Per-call generation becomes ~1–3 tokens plus EOS (512 max is irrelevant).
3. **Advantage density.** One action per step means each step contributes ~1–3 policy tokens. Token-mean then weights long episodes more heavily, as ActiveVLN does, but with ~3× more decisions per episode. Keep token-mean for faithfulness, and optionally log a sequence-mean variant.
4. **Context.** ActiveVLN conditions on the full history (every frame since the start). SimpleMem uses a KV window of 8 observation/action groups.
   - Log-probs for training must be recomputed **under exactly the same windowed context** that generated each action. This means a streaming/replay forward with the same eviction, not one long concatenated sequence.
   - ActiveVLN's "one long sequence + action_mask" assembly cannot be reused as is. Either replay step by step with cached windows, or build per-step (window, action) training items that share the episode advantage.
5. **Invalid output.**
   - ActiveVLN terminates on any malformed output with R=0.
   - SimpleMem's discrete text actions (`MOVE_FORWARD/TURN_LEFT/TURN_RIGHT/STOP`) should keep "unparseable → terminate, R=0" for faithfulness. Note that ActiveVLN's *evaluation* instead substitutes a random action.
6. **Prompt and tokenizer.** Replace the Qwen2.5 template, the hard-coded token ids (151644/77091/198/151645), the `_strip_system_block` assumption, and the 88-token-per-frame resize (`max_pixels=76800`) with SimpleMem's own observation format. Do not import ActiveVLN's prompt.
7. **Simulator protocol.**
   - ActiveVLN uses habitat **0.1.7**, RGB 640×480, **HFOV 90**, camera at 1.25 m, ALLOW_SLIDING True.
   - SimpleMem's evaluator uses habitat-lab **0.2.4** with **HFOV 79** (`configs/eval/vln_r2r.yaml`). Camera height and sliding defaults in SimpleMem's 0.2.4 config are UNVERIFIED.
   - Use the SimpleMem protocol for RL rollouts so that training matches SimpleMem eval, and **do not compare SR across the two protocols**.
   - The success threshold (3.0 m), step size (0.25 m) and turn (15°) match.
8. **Vision tower.** ActiveVLN updates *all* parameters in RL, including the ViT. Decide whether SimpleMem RL trains the vision tower; full fine-tuning is the faithful choice, at higher memory cost.
9. **Infrastructure.** ActiveVLN's HTTP/Ray server, actor pool, scene caching and pre-loading are throughput features, not algorithmic ones. They can be replaced by SimpleMem's in-process habitat env workers. Keep "identical start state for the 4 group members", for example by resetting to the same episode.
10. **Evaluation comparison.**
    - ActiveVLN eval uses sampled decoding (T=0.2, top_p=0.8), forced stops (400 primitives, 25 non-progress primitives), and 120 model calls.
    - SimpleMem's protocol differs, so evaluate SimpleMem baseline vs SimpleMem+GRPO under SimpleMem's own identical protocol.
    - Quote ActiveVLN's 38.5 → 50.1 SR (Qwen2.5-VL-3B, habitat 0.1.7, HFOV 90) only as context.
