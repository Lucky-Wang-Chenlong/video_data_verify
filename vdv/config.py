"""Build the lite_model Hydra config of one video_data_verify run.

The model, loss, optimizer and LR schedule come unchanged from lite_model's
``configs/depth/train_data_scaling_lite.yaml`` (identical to the HF/WDS lite
config the released checkpoint was trained with). This module only replaces:

  * trainer.data.train  -> VDVMixDataset: standard CA-1M + custom data at ``ratio``
  * trainer.data.val    -> one GT-box-prompted clip set per test split (ca1m, custom)
  * trainer.meters / loss keys for those sets, trainer target -> VDVTrainer
  * epochs / epoch length / LR / seed, init checkpoint, output dir, logging

Experiment design (README.md): A0 zero-shot, A1 control (ratio 0), A2 treatment
(ratio > 0). A1 and A2 share epochs, clips per epoch, LR schedule and seed, so
differences on the custom test set come from the custom data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import OmegaConf

BASE_CONFIG = "configs/depth/train_data_scaling_lite.yaml"

# ------------------------------------------------------------------ defaults
EPOCHS = 5
SAMPLES_PER_EPOCH = 1600     # global training clips per epoch (1 clip per GPU step)
WORKERS_PER_GPU = 4
BASE_LR = 3.0e-5             # heads (upstream 1e-4 degrades the converged checkpoint); backbone + spatial encoder: upstream vision_lr 1e-7
SEED = 123
VAL_CLIP_FRAMES = 8          # upstream CA-1M val protocol: 8 consecutive frames ...
VAL_MAX_OBJECTS = 10         # ... first 10 valid objects of the first frame
VAL_STRIDE = 8               # non-overlapping clips


@dataclass
class RunSettings:
    out: str
    std_root: str
    custom_root: Optional[str] = None
    ratio: float = 0.2
    mode: str = "train"                    # train | val
    init_ckpt: Optional[str] = None
    epochs: int = EPOCHS
    samples_per_epoch: int = SAMPLES_PER_EPOCH
    gpus: int = 1
    workers: int = WORKERS_PER_GPU
    lr: float = BASE_LR
    seed: int = SEED
    custom_train_split: str = "train"
    custom_test_split: str = "test"
    depth_scale: float = 1000.0
    val_sets: List[str] = field(default_factory=lambda: ["all"])
    val_stride: int = VAL_STRIDE
    val_every: int = 1                     # validate every N epochs (and after the last)
    max_val_clips: Optional[int] = None    # per GPU, for quick checks
    overrides: List[str] = field(default_factory=list)


def _val_collate(key):
    return {
        "_target_": "sam3.train.data.collator.collate_fn_api", "_partial_": True,
        "repeats": "${scratch.hybrid_repeats}", "dict_key": key,
        "with_seg_masks": True, "enforce_single_datapoint_batch": True,
    }


def _clip_options(training: bool, depth_scale: float):
    """Loader options shared with lite_model's CA-1M train / val configs."""
    common = {
        "load_segmentation": True, "max_train_queries": 100000, "max_val_queries": 100000,
        "default_scale": depth_scale, "predict_metric": True, "norm_scale": "${scratch.norm_scale}",
        "resolution": "${scratch.resolution}", "spatial_type": "depthanythingv3",
        "spatial_resolution": "${scratch.resolution}",
        "min_crop_visible_ratio": "${scratch.min_crop_visible_ratio}", "training": training,
    }
    if training:
        common.update(
            transforms="${scratch.video_transforms_train}", shuffle=True,
            video_split_per_frame="${scratch.video_split_per_frame}",
            frame_num_range="${scratch.frame_num_range}",
            frame_sample_gap_range="${scratch.frame_sample_gap_range}",
            max_num_objects="${scratch.max_num_objects}", resolution_aug_scale=[0.8, 1.2])
    else:
        common.update(
            transforms="${scratch.video_transforms_val}", shuffle=False,
            video_split_per_frame=VAL_CLIP_FRAMES, frame_num_range=[VAL_CLIP_FRAMES] * 2,
            frame_sample_gap_range=[1, 1], max_num_objects=VAL_MAX_OBJECTS, use_extrinsic=True)
    return common


def load_standard(std: str) -> dict:
    """The standard data: training sources (with relative weights) and test sets.

    ``std`` is either a dataset root in the vdv format (CA-1M: its train split
    trains, its test split is the ``ca1m`` test set) or a manifest JSON

        {"train": [{"name", "root", "source", "split", "weight", "depth_scale"?}, ...],
         "test":  [{"key", "root", "source", "split", "depth_scale"?, "stride"?}, ...]}

    A test set's "stride" (frames between clip starts) overrides --val-stride.

    whose roots may use $VDV_DATA or be relative to the manifest file.
    """
    std = os.path.abspath(os.path.expanduser(std))
    if os.path.isdir(std):
        return {"train": [{"name": "CA-1M", "root": std, "source": "CA-1M", "split": "train",
                           "weight": 1.0}],
                "test": [{"key": "ca1m", "root": std, "source": "CA-1M", "split": "test"}]}
    import json

    with open(std) as f:
        manifest = json.load(f)
    base = os.path.dirname(std)
    for entry in manifest["train"] + manifest["test"]:
        root = os.path.expandvars(entry["root"])
        if "$" in root:
            raise ValueError(f"unset variable in standard manifest root: {entry['root']}")
        entry["root"] = os.path.normpath(os.path.join(base, root))
    return manifest


def _val_set(key, root, source, split, s: RunSettings, depth_scale=None, stride=None):
    if depth_scale is None:
        depth_scale = 1000.0 if source == "CA-1M" else s.depth_scale
    options = _clip_options(False, depth_scale)
    return {
        "_target_": "sam3.train.data.torch_dataset.TorchDataset",
        "dataset": {"_target_": "vdv.data.VDVVideoDataset", "root": root, "source": source,
                    "split": split, "slide_stride": stride or s.val_stride, **options},
        "shuffle": False, "batch_size": 1, "num_workers": s.workers, "pin_memory": True,
        "drop_last": False, "prefetch_factor": 2, "collate_fn": _val_collate(key),
    }


def build_config(s: RunSettings):
    from hydra import compose, initialize_config_module

    with initialize_config_module("sam3.train", version_base="1.2"):
        cfg = compose(config_name=BASE_CONFIG)
    OmegaConf.set_struct(cfg, False)
    out = os.path.abspath(s.out)
    use_custom_train = s.custom_root is not None and s.ratio > 0
    use_std_train = s.ratio < 1

    # ---------------- paths / budget / optimisation
    cfg.paths = {"dump_file_name": "video_data_verify", "experiment_log_dir": out,
                 "sam3_checkpoint": s.init_ckpt}
    cfg.launcher.experiment_log_dir = out
    cfg.launcher.gpus_per_node = s.gpus
    cfg.scratch.max_data_epochs = s.epochs
    cfg.scratch.num_train_workers = s.workers
    cfg.scratch.samples_per_worker_per_epoch = max(1, s.samples_per_epoch // (s.gpus * s.workers))
    cfg.scratch.lr = s.lr
    t = cfg.trainer
    t._target_ = "vdv.trainer.VDVTrainer"
    t.mode = s.mode
    t.seed_value = s.seed
    t.val_epoch_freq = s.val_every
    t.skip_first_val = False
    t.max_val_iters = s.max_val_clips
    t.model.use_fa3 = False
    t.model.checkpoint_path = s.init_ckpt
    t.checkpoint.save_dir = os.path.join(out, "checkpoints")
    t.checkpoint.save_freq = 0
    t.logging.log_dir = os.path.join(out, "logs")
    t.logging.tensorboard_writer.log_dir = os.path.join(out, "tensorboard")
    t.logging.wandb_writer.enabled = False

    # ---------------- training data
    std = load_standard(s.std_root)
    sources = {}
    if use_std_train:
        total = sum(float(e["weight"]) for e in std["train"])
        for e in std["train"]:
            sources[e["name"]] = {"root": e["root"], "source": e["source"], "split": e["split"],
                                  "weight": (1.0 - s.ratio) * float(e["weight"]) / total,
                                  "options": {"default_scale": float(e.get("depth_scale", 1000.0))}}
    if use_custom_train:
        sources["custom"] = {"root": s.custom_root, "source": "VDV-Custom",
                             "split": s.custom_train_split, "weight": s.ratio,
                             "options": {"default_scale": s.depth_scale}}
    train = OmegaConf.to_container(t.data.train, resolve=False)
    train["dataset"] = {"_target_": "vdv.data.VDVMixDataset", "sources": sources,
                        "num_workers": "${scratch.num_train_workers}",
                        "samples_per_worker_per_epoch": "${scratch.samples_per_worker_per_epoch}",
                        **_clip_options(True, 1000.0)}
    train["dataset"].pop("default_scale")

    # ---------------- validation sets (GT 2D box prompts)
    # val_sets: test-set keys; "standard" = every standard test set, "all" = standard + custom
    wanted = set(s.val_sets)
    if "all" in wanted:
        wanted |= {"standard", "custom"}
    val = {}
    for e in std["test"]:
        if e["key"] in wanted or "standard" in wanted:
            val[e["key"]] = _val_set(e["key"], e["root"], e["source"], e["split"], s,
                                     float(e.get("depth_scale", 1000.0)), e.get("stride"))
    if "custom" in wanted and s.custom_root is not None:
        val["custom"] = _val_set("custom", s.custom_root, "VDV-Custom", s.custom_test_split, s)
    t.data = {"train": train, "val": val or None}

    loss_all = OmegaConf.to_container(t.loss["all"], resolve=False)
    t.loss = {"all": loss_all, **{k: loss_all for k in val},
              "default": {"_target_": "sam3.train.loss.sam3_loss.DummyLoss"}}
    t.meters = {"train": None, "val": {k: {"box3d": {"_target_": "vdv.metrics.Box3DMeter",
                                                      "norm_scale": "${scratch.norm_scale}"}}
                                       for k in val} or None}

    # Extra "dotted.key=yaml_value" overrides, e.g. scratch.frame_num_range=[2,4].
    import yaml

    for item in s.overrides:
        key, value = item.split("=", 1)
        OmegaConf.update(cfg, key.lstrip("+"), yaml.safe_load(value), force_add=True)
    return cfg
