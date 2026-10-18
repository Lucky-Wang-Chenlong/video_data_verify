# video_data_verify

English | [中文](README_zh.md)

Check whether **your own generated video 3D detection / tracking data is correct** by finetuning on it. This is the video version of [data_verify](../../data_verify/data_verify).

Method: start from the released SpatialEncoder v2 Small (lite) checkpoint, finetune on the standard data (10k frames sampled uniformly from the SpatialEncoder training set) mixed with your data at a given ratio, then check that your test set improves while the standard test set (CA-1M test) does not regress. The model code is [lite_model](https://github.com/qic999/spatial_encoder_v2_small); this repo only holds its own code and does not modify lite_model.

```
1. Install                 ./setup_env.sh
2. Download standard data  scripts/download_data.sh --accept-license
3. Prepare your dataset    put it in one directory in the format of vdv/README.md; check it with scripts/validate.sh
4. Train                   scripts/train.sh --custom <dataset dir> --ratio <mixing ratio>
5. Read the output         --out: results/summary.md, metrics/ (losses, every validation)
```

## Model and evaluation

The lite model is a **GT-2D-box-prompted video 3D box estimator and tracker**: each object gets its GT 2D box in the first frame of a clip, and the model outputs its 3D box in every following frame (plus depth maps and camera poses). It has no category recognition / text prompt, so there is no AP and no box / text evaluation modes as in data_verify.

Evaluation: every test video is cut into non-overlapping 8-frame clips; in each clip up to 10 objects are prompted with their first-frame GT 2D boxes, and every visible (object, frame) slot is scored by the 3D IoU between the predicted and the GT box. Reported:

| metric | meaning |
|---|---|
| `miou` | mean 3D IoU (%) |
| `acc25` / `acc50` / `acc10` | fraction of slots with 3D IoU ≥ 0.25 / 0.5 / 0.1 (%) |
| `trans_err` / `depth_err` | 3D box center error / depth error (m) |
| `yaw_err` | orientation error about the vertical (deg), folded to 0–45° by the cuboid's footprint symmetry (90° turn + width/length swap is the same box); `yaw_err_raw` is lite_model's unfolded value, which depends on each dataset's horizontal axis order (≈90° for PCA-fitted boxes even when they are right); `scale_err` = 1 − size IoU |
| `n` | number of evaluated (object, frame) slots |

Each is reported overall and **per category**. The standard test set is CA-1M test (2 CA-1M val captures); most of its boxes are labeled `object`, so it is mainly the overall regression check. Whether your target categories improve is read from your own test set.

---

## 1. Install

```bash
mkdir -p <workspace> && cd <workspace>     # workspace: lite_model, env, weights, data and outputs live here
git clone <this repo> repo
hf auth login                              # the weights are in the gated HF repo SpatialCoding/Data_verification
cd repo && ./setup_env.sh                  # log: <workspace>/envs/setup_env.log
```

This:
- clones lite_model next to this repo;
- creates the venv `../envs/lite` with the system Python 3.10 and installs torch 2.10 (cu128), Depth-Anything-3 etc. following the lite_model README;
- downloads the released lite checkpoint to `../ckpts/se_small_lite_wds518_e100/checkpoint.pt` (2.4 GB; it already holds all SAM3 tracker weights, `checkpoint_64.pt` is not needed).

Hardware: Linux + NVIDIA GPUs, about 4 GB GPU memory per GPU at 1 clip per GPU.

## 2. Download the standard data

```bash
scripts/download_data.sh --accept-license               # into ../data by default
scripts/download_data.sh /path/to/data --accept-license # or elsewhere (then export VDV_DATA=/path/to/data)
```

The standard data is the lite model's own training set (HF `qicq1c/spatialencoder_full`: train splits of CA-1M + ADT + Hypersim, about 2.17M frames), **sampled uniformly by frame down to 10k frames**: a sequence is drawn proportionally to its frame count, then a non-overlapping window of 100 consecutive frames inside it (the model trains on video clips), 100 windows in total, so source proportions match the full set. The sample uses a fixed seed and is stored in `data/standard_10k_plan.json`, so every build yields the same data; it is defined in `data/standard_10k.json`.

CA-1M is released under CC-BY-NC-ND 4.0 (non-commercial); `--accept-license` accepts it. Finished steps are skipped, so re-running resumes. About 90 GB of archives are downloaded (ADT ships one 6.7 GB tar per sequence), but only the sampled frames are kept and the archives are deleted, leaving about 12 GB:

| content | path (under DATA_DIR) | use |
|---|---|---|
| uniform 9,916-frame sample: CA-1M 9,216 frames (93 captures; one capture holds fewer frames than its index entry, so its window has 16), ADT 500 (5 sequences), Hypersim 200 (2 camera sequences) | `standard_10k/{ca1m,adt,hypersim}/` | standard training data, mixed by frame count |
| 2 CA-1M val captures (2,131 frames, 270 test clips, see `data/ca1m_test.txt`) | `ca1m/raw/`, converted `ca1m/vdv_test/` | forgetting check |
| `cushion` boxes of 3 more CA-1M train captures (train 361 frames, test 349 frames / 45 clips) | `ca1m/raw_toy/`, converted `custom/toy/` | toy custom set: format example, smoke tests |

## 3. Prepare your dataset

Format: **[vdv/README.md](vdv/README.md)**:

```
/path/to/mydata/
├── index.json        [[scene json, n_frames, "train"|"test"], ...]
├── scenes/*.json     per video: frames (image, depth, intrinsics, camera pose) and objects
│                     (per-frame 3D and 2D boxes, one track id across frames)
└── ...               image and depth files
```

Validate first. Fix every ERROR, then check in the renders (3D boxes on consecutive frames) that the boxes fit the objects:

```bash
scripts/validate.sh --custom /path/to/mydata    # renders in ../outputs/video_data_verify/vis/<dataset>/<split>/
```

Besides the schema it checks that projected 3D boxes agree with `camera_box_2d_proj` (axes, units, intrinsics), that the box's local +Y points down (gravity), depth units and depth vs box centers, world-space drift of static objects (pose direction, units), and how many objects pass lite_model's object filter.

Then run the overfit check: train on your train split only and evaluate on it. mIoU should rise clearly; if it stays flat, your annotations are inconsistent:

```bash
scripts/train.sh --custom /path/to/mydata --overfit
```

## 4. Train

```bash
scripts/train.sh --custom /path/to/mydata --ratio 0.2 --out ../outputs/video_data_verify/mydata_r0.2
```

`--ratio` is the fraction of training clips drawn from your data; the rest come from CA-1M.

| `--ratio` | meaning |
|---|---|
| `0` | standard data only: the control (A1) |
| `0.2` | 80% standard + 20% yours (A2, default) |
| `1` | your data only |

Runs with different ratios share epochs, clips per epoch, LR schedule and seed, so differences come from the data only.

**Common options** (full list: `scripts/train.sh -h`):

| option | default | meaning |
|---|---|---|
| `--gpus N` | 8 | number of GPUs (select them with `CUDA_VISIBLE_DEVICES`) |
| `--epochs N` | 5 | cosine LR decay to 1/100 over all epochs |
| `--samples N` | 1600 | training clips per epoch in total; 200 steps per epoch on 8 GPUs |
| `--lr X` | 3e-5 | head LR (upstream trains with 1e-4, which degrades the converged checkpoint here); the SAM3 backbone and spatial encoder keep the upstream 1e-7 |
| `--train-split` / `--test-split` | `train` / `test` | split tags in `index.json` |
| `--depth-scale S` | 1000 | 16-bit PNG depth units per meter |
| `--val-every N` | 1 | validate every N epochs; always after the last one |
| `key=value` | | other lite_model config overrides, e.g. `scratch.frame_num_range=[2,4]` |

Every validation covers the standard test set (CA-1M test) and your test set. **The validation after the last epoch uses the final weights and is the result of the run** (`results/summary.md`); no separate evaluation is needed. Re-running with the same `--out` resumes an interrupted run.

**Full comparison in one go**: zero-shot baseline plus one training run per ratio, then one table:

```bash
nohup scripts/compare.sh --custom /path/to/mydata --ratios "0 0.2" > compare.log 2>&1 &
```

**Evaluate a checkpoint on its own**:

```bash
scripts/eval.sh --ckpt released --custom /path/to/mydata                       # zero-shot (A0)
scripts/eval.sh --ckpt <run>/checkpoints/checkpoint.pt --custom /path/to/mydata
scripts/eval.sh --ckpt <run>/checkpoints/checkpoint.pt               # standard test sets only (--no-standard: your test set only)
```

## 5. Output

```
<out>/
├── checkpoints/checkpoint.pt   final weights (2.4 GB; also the resume point)
├── metrics/
│   ├── train.csv               every 10 steps: total loss, loss terms, learning rate
│   ├── val.csv                 one row per validation x test set: miou, acc10/25/50, errors, n
│   └── val_per_category.csv    the same per category
├── results/
│   ├── summary.md              final results (overall + per category)
│   └── summary.json
├── logs/                       train.log / eval.log (full output), log.txt, train_stats.json, val_stats.json
├── tensorboard/                tensorboard --logdir <out>
└── config.yaml                 complete lite_model config of the run
```

`compare.sh` writes one subdirectory per run and a table `compare.md`:

```
| run      | custom miou | custom acc25 | custom acc50 | ca1m miou | ca1m acc25 | ca1m acc50 |
| zeroshot | ...                                                                 |   ← A0
| ratio0   | ...                                                                 |   ← A1 (control)
| ratio0.2 | ...                                                                 |   ← A2
```

followed by a per-category mIoU table on your test set.

**How to read it**: if ratio0.2 is clearly above ratio0 on your test set while both are on par on the CA-1M test set, the data is right and the model learns from it. If ratio0.2 is worse on your test set, or CA-1M drops clearly, first look at the WARNs of `validate.sh` (pose direction, units, 2D/3D box disagreement).

To re-export metrics or compare any runs:

```bash
source scripts/env.sh
python $VDV/tools/export_metrics.py run <out>
python $VDV/tools/export_metrics.py compare <run1> <run2> ... --to table.md
```

---

## Verified results (released lite checkpoint, 8×H100)

**Zero-shot** (`scripts/eval.sh --ckpt released --custom data/custom/toy`, about 1 minute):

| test set | miou | acc10 | acc25 | acc50 | trans_err (m) | yaw_err (°) | n |
|---|---|---|---|---|---|---|---|
| CA-1M test (270 clips) | 14.7 | 45.4 | 26.3 | 4.7 | 0.489 | 31.5 | 4727 |
| toy test (cushion, 45 clips) | 17.8 | 55.6 | 34.9 | 4.9 | 0.342 | 17.3 | 430 |

**Toy overfit** (`scripts/train.sh --custom data/custom/toy --overfit`, 10 epochs × 800 clips, about 20 minutes): on the toy train split mIoU 24.6 → 46.2, acc25 46 → 81, acc50 12.5 → 48, rising almost monotonically: on consistent labels the model learns and the metrics show it.

**Learning rate**: with the upstream 1e-4 a short smoke test (`--epochs 2 --samples 320`) drops CA-1M from 14.7 to 13.5; with 3e-5 and the full budget (5 epochs × 1600 clips, standard data = the earlier CA-1M-only version) CA-1M no longer regresses, hence the 3e-5 default:

| run (lr 3e-5) | toy miou | toy acc25 | toy acc50 | CA-1M miou | CA-1M acc25 | CA-1M acc50 |
|---|---|---|---|---|---|---|
| zero-shot | 17.8 | 34.9 | 4.9 | 14.7 | 26.3 | 4.7 |
| ratio0 | 21.5 | 42.1 | 11.6 | 14.7 | 26.7 | 5.6 |
| ratio0.2 | 19.2 | 35.6 | 8.1 | 14.4 | 26.3 | 5.2 |

ratio0.2 below ratio0 on the toy set does not mean the toy labels are wrong (they are the official CA-1M labels): the toy train split is 361 frames of a single capture, from which ratio0.2 draws about 1600 clips over 5 epochs, i.e. it overfits one scene, while the CA-1M train set already holds cushions from many scenes. The toy test set has only 45 clips and moves by about 1 point between epochs. Use the toy set for smoke tests only; a real custom set should span many scenes, ideally with categories the standard data lacks or the zero-shot model handles badly.

---

## Layout

```
<workspace>/
├── lite_model/           upstream repo, unmodified
├── envs/lite/            environment
├── ckpts/                released lite checkpoint
├── data/                 standard and toy data (default VDV_DATA)
├── outputs/video_data_verify/  default output parent
└── repo/                 ← this repo
    ├── setup_env.sh
    ├── data/standard_10k.json       standard data: training sources + weights, test set (default VDV_STD)
    ├── data/standard_10k_plan.json  the 100 uniformly sampled windows (source, sequence, start frame)
    ├── data/ca1m_test.txt           the 2 CA-1M val captures of the test set (ca1m_10k.txt: the earlier CA-1M-only standard set)
    ├── scripts/
    │   ├── env.sh            sourced by the others: PATH, PYTHONPATH, default paths
    │   └── download_data.sh, validate.sh, train.sh, eval.sh, compare.sh
    ├── vdv/
    │   ├── README.md         dataset format
    │   ├── data.py           datasets (on lite_model's generic video loader) and mixing
    │   ├── config.py         replaces data, validation, metrics and budget in the lite_model config
    │   ├── trainer.py        Trainer with several validation sets
    │   ├── metrics.py        3D IoU metrics (overall + per category)
    │   └── run.py            train / eval entry point (torchrun)
    └── tools/                standard-set sampling (sample_standard.py), CA-1M / Infinigen conversion, toy set, validation, dataloader check, metric export
```

**How it runs**: scripts source `scripts/env.sh` and launch `python -m vdv.run` with `torchrun`. `vdv/config.py` loads lite_model's `configs/depth/train_data_scaling_lite.yaml` (model, loss and optimizer identical to the HF/WDS lite config the released weights were trained with), replaces only training data, validation sets, metrics, epoch budget, LR, initial weights and output directory, and hands it to lite_model's own `run_training`.

## Differences from upstream and caveats

- **Standard data through the generic loader**: CA-1M is converted to the same format as custom data and read by lite_model's `ObjectDetectionDataset` (still with CA-1M's dataset id 0), not by `CA1MTrainIterableDataset`, which needs pickles + the original tars. The difference: the native CA-1M loader resamples the 256×192 depth to the RGB image through the intrinsics, the generic loader resizes it; CA-1M's depth intrinsics are exactly the RGB ones / 4, so the two differ by under 1.5 RGB pixels. The `T_gravity` of the native loader is not used by any loss or metric. ADT and Hypersim already come as generic scene JSONs on HF; only the sampled frames are cut out, and they are read under the source names `ADT` and `hyperism` so lite_model applies its axis and pose conventions.
- **Evaluation protocol**: lite_model's CA-1M validation settings (8 frames, gap 1, first 10 objects), but with non-overlapping clips (stride 8, 270 CA-1M test clips) and unscored padding, so every clip counts once.
- **Upstream evaluations turned off**: ScanNet val88, WildDet3D 10-dataset, HF source probes and the CA-1M 1000-case reference all need cluster data; W&B is disabled by default.
- **Learning rate**: cosine decay to 1/100 as upstream, but the head LR starts at 3e-5 instead of 1e-4 (backbone and spatial encoder 1e-7 as upstream), over fewer epochs, no warmup.
- **`use_fa3=false`**: FlashAttention 3 is not required.
- **Disk**: each run keeps one 2.4 GB checkpoint (the latest).
