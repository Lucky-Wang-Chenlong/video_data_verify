"""Datasets for video_data_verify, built on lite_model's generic video loader.

Both the standard data (CA-1M subset) and the user's data use one on-disk
format (see vdv/README.md):

    <root>/index.json            [[scene json (relative to root), n_frames, split], ...]
    <root>/scenes/<scene>.json   scene_id / frames / objects (lite_model scene JSON)
    <root>/...                   assets; rgb_path / depth_path are relative to root

and are read by sam3.train.data.object_detection_dataset.ObjectDetectionDataset
without modifying lite_model:

  * the standard set is read under the source name "CA-1M" (its dataset id 0;
    the generic loader has no CA-1M specific branch),
  * the user's set under "VDV-Custom", registered below. Frames of a custom
    set may omit depth_path (the depth loss then ignores them).

Importing this module performs the registration, and every dataset class here
lives in this module, so spawned DataLoader workers re-run it on unpickling.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import tempfile
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch.utils.data import IterableDataset

from sam3.train.data import object_detection_dataset as _odd
from sam3.train.data.object_detection_dataset import DATASET_IDS, ObjectDetectionDataset
from sam3.train.utils.distributed import get_world_size

STANDARD_SOURCE = "CA-1M"
CUSTOM_SOURCE = "VDV-Custom"
CUSTOM_DATASET_ID = 100

DATASET_IDS.setdefault(CUSTOM_SOURCE, CUSTOM_DATASET_ID)
if CUSTOM_SOURCE not in _odd.DATASETS_WITHOUT_DENSE_DEPTH:
    _odd.DATASETS_WITHOUT_DENSE_DEPTH = _odd.DATASETS_WITHOUT_DENSE_DEPTH | {CUSTOM_SOURCE}


def read_index(root: str) -> list:
    with open(os.path.join(root, "index.json")) as f:
        return json.load(f)


class VDVVideoDataset(ObjectDetectionDataset):
    """One split of a dataset in the vdv format.

    Training: iterable, clips sampled as in lite_model (random clip length and
    frame gap, random objects). Evaluation: map-style, every clip of
    ``video_split_per_frame`` frames at stride ``slide_stride``, padded to a
    multiple of the world size with empty samples (skipped by the trainer) so
    no clip is counted twice.
    """

    def __init__(self, root: str, source: str, split: str, **kwargs):
        root = os.path.abspath(os.path.expanduser(root))
        if source not in DATASET_IDS:
            raise ValueError(f"unknown source {source}")
        # ObjectDetectionDataset reads <meta_dir>/<source>.json.
        meta_dir = tempfile.mkdtemp(prefix="vdv_meta_")
        try:
            shutil.copy(os.path.join(root, "index.json"), os.path.join(meta_dir, f"{source}.json"))
            super().__init__(meta_dir=meta_dir, dataset_name=source, base_dir=root,
                             meta_base_dir=root, splits=[split], **kwargs)
        finally:
            shutil.rmtree(meta_dir, ignore_errors=True)
        self.root = root
        self.split = split
        self.num_real_clips = len(self.meta_file_paths)
        if not self.training:
            world = get_world_size()
            pad = (-len(self.meta_file_paths)) % world
            self.meta_file_paths = self.meta_file_paths + ["padding"] * pad

    def __getitem__(self, idx: int):
        if self.meta_file_paths[idx % len(self.meta_file_paths)] == "padding":
            return {}
        sample = super().__getitem__(idx)
        return {} if sample is None and not self.training else sample


class VDVMixDataset(IterableDataset):
    """Weighted mixture of training sources, one source drawn per clip.

    Same epoch definition as lite_model's AggregateDataset:
    ``samples_per_worker_per_epoch * num_workers`` clips per rank. Sources with
    weight 0 are never instantiated, so a control run never reads custom data.
    """

    def __init__(self, sources: Dict[str, dict], num_workers: int,
                 samples_per_worker_per_epoch: int, **common):
        super().__init__()
        self.length = int(samples_per_worker_per_epoch) * int(num_workers)
        self.names, self.weights, self.datasets = [], [], {}
        for name, spec in sources.items():
            weight = float(spec["weight"])
            if weight <= 0:
                continue
            if not math.isfinite(weight):
                raise ValueError(f"bad weight for {name}: {weight}")
            cfg = dict(common)
            cfg.update(spec.get("options") or {})
            self.datasets[name] = VDVVideoDataset(
                root=spec["root"], source=spec["source"], split=spec.get("split", "train"),
                num_workers=num_workers, samples_per_worker_per_epoch=samples_per_worker_per_epoch,
                **cfg)
            self.names.append(name)
            self.weights.append(weight)
        if not self.names:
            raise ValueError("VDVMixDataset needs at least one source with weight > 0")

    def __len__(self):
        return self.length

    def __iter__(self) -> Iterable:
        info = torch.utils.data.get_worker_info()
        target = max(len(self) // info.num_workers, 1) if info is not None else len(self)
        iters = {n: iter(self.datasets[n]) for n in self.names}
        for _ in range(target):
            name = random.choices(self.names, weights=self.weights)[0]
            try:
                sample = next(iters[name])
            except StopIteration:
                iters[name] = iter(self.datasets[name])
                sample = next(iters[name])
            sample.reference_payload["aggregate_source"] = name
            yield sample
