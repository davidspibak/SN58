# vector-train

Fine-tunes `vector_v1.1` weights (Robotensor Vector, Bittensor SN58), starting from the king's, and
exports a `model.safetensors` that passes the validator's own check.

The architecture is pinned, so a submission can only change the numbers in the same 753 tensors.
Every past champion did exactly that: the U-Net and decoder moved 4–6%, CLIP about 0.4%, and the
normalizer not at all. This pipeline follows that pattern.

## On the GPU box

```bash
# 0. environments (copy this folder next to where you run it first)
bash vector-train/scripts/setup_gpu_box.sh
source .venv-train/bin/activate

# 1. data: 8,000 randomized runs (63.6 GB) + the 800 clean ones
bash vector-train/scripts/fetch_data.sh data/randomized franka_randomized_500
bash vector-train/scripts/fetch_data.sh data/clean franka_clean_50

# 2. check the pipeline against the validator's code (a few minutes)
vector-train selftest --base king/model.safetensors --data data/randomized --data data/clean --device cuda

# 3. frozen camera features, once (about 2.4M images; fp32 like inference)
vector-train cache --base king/model.safetensors --data data/randomized --data data/clean --out cache/

# 4. train (weak tasks weighted up); watch run1/log.jsonl. Every 2,000 steps a validated snapshot
#    run1/ema-NNNNNNN.safetensors (about 2 GB) is kept: 10 per 20,000-step run
vector-train train --base king/model.safetensors --cache cache/ --run run1 \
    --steps 20000 --batch 64 --lr 1e-5 --task-weight move_stapler_pad=2 --task-weight place_dual_shoes=2

# 5. rank every snapshot of one or more runs by validation loss, against the king's
vector-train snapshots --run run1 run2 run3

# 6. export the chosen ones: exports/<run>-<step>/submission/model.safetensors each, checked,
#    plus exports/exports.json listing them all (it refuses two with identical weights)
vector-train export --base king/model.safetensors --out exports/ \
    --snapshot run1/ema-0012000.safetensors run1/ema-0020000.safetensors run2/ema-0016000.safetensors

# 7. a local duel against the king with the validator's engine (needs the simulator)
export POLICY_PYTHON=$PWD/.venv-train/bin/python ORCH=$PWD/.venv-orch/bin/vector-orchestrator
export ROBOTWIN_BENCH_ROOT=/path/to/RoboTwin-Vector ROBOTWIN_BENCH_PYTHON=/path/to/envs/robotwin/bin/python
bash vector-train/scripts/local_duel.sh king/model.safetensors \
    exports/run1-0020000/submission/model.safetensors duels/run1-0020000 32
```

Submit only when local duels show a clear lead (about +6 points over 500+ units), with
`attune miner check --dir exports/<name>/submission` and then `attune miner submit ... --network finney --netuid 58`.

## Submitting several models

The validator takes one submission per hotkey and refuses weights byte-identical to an earlier
commitment's; nothing else limits several hotkeys. So N models need N registered hotkeys and N
Hugging Face repos (one `model.safetensors` each, kept public: a repo that disappears is dropped,
and a champion's repo that disappears stops being paid).

- **Pick from several runs.** Snapshots of one run are close relatives: they mostly win or lose
  together. Runs with other seeds (`--seed`), task weights or learning rates give candidates that
  differ more. Train each in its own `--run` directory and rank them all with `snapshots`.
- **Your own crown is the next bar.** The queue runs oldest commitment first. Once one of yours is
  king, each of your later ones must beat it by 3 points, which a near-twin almost never does, and
  its hotkey's one submission is spent.

## Layout

| File | What it does |
|---|---|
| `src/vector_train/episodes.py` | Finds and reads RoboTwin HDF5 episodes; converts rows with `vector_runtime.demo` |
| `src/vector_train/features.py` | Builds the frozen-CLIP feature cache |
| `src/vector_train/batches.py` | Draws (prompt episode, target episode, step) samples onto the GPU |
| `src/vector_train/model.py` | The training forward and diffusion loss, mirroring `Net` |
| `src/vector_train/train.py` | AdamW, warmup + cosine, bf16, EMA, validation, snapshots (all kept), resume |
| `src/vector_train/snapshots.py` | Lists every run's snapshots with their validation loss against the king's |
| `src/vector_train/export.py` | Snapshots + base → `model.safetensors` each, `vector_runtime.check`, drift report, no two identical |
| `src/vector_train/selftest.py` | Ten checks against the validator's code |
| `scripts/` | GPU box setup, dataset download, local duel |

## Choices that matter

- **Colours.** RoboTwin stored its frames with OpenCV straight from RGB arrays. They have to be decoded
  with `cv2.imdecode` and used as they come. Any other decoder swaps red and blue, which the mat colours
  named in `scene_info.json` confirm: 27/27 clean and 155/157 decided randomized episodes.
- **Same code as the validator.** Row conversions (rot6d, proprioception, actions), resizing and
  normalization are `vector_runtime`'s own functions, and the environment pins its versions.
- **Frozen CLIP, cached.** Each camera image enters the network only as one 768-number CLS token.
  With the ViTs frozen, those numbers are computed once (about 3.7 GB for everything), so training
  runs the decoder and U-Net only. This is much faster, but it means no image augmentation.
- **Prompts as the duel builds them.** One frame every 15 steps and every action, padded as
  `demo.prompt` pads. Prompts of different lengths share a batch, and the padding is masked out of
  the cross-attention, so each sample is computed exactly as inference computes it alone.
- **Targets.** History frames t-1 and t (frame 0 repeated at the start), the next 16 actions, and the
  last action repeated past the end of an episode.
- **Untouched.** The normalizer (`norm.*` buffers) and CLIP (`vision.*`).

## Not covered yet

- Training CLIP itself, which would need live images and augmentation.
- `scripts/local_duel.sh` is written from the orchestrator's CLI but has not been run here: it needs
  the simulator.
- Level-3 training data must be collected with RoboTwin-Vector's collector at `level: 3`. The
  pipeline reads its HDF5 output like any other data root.
