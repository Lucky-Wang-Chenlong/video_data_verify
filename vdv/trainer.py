"""lite_model Trainer with several named validation sets.

``trainer.data.val`` is a mapping ``{key: TorchDataset config}`` (e.g.
``ca1m`` and ``custom``). Each set's collate_fn uses its key as ``dict_key``,
so the upstream trainer routes every batch to ``trainer.loss[key]`` and
``trainer.meters.val[key]``; the sets are simply run one after another inside
a single validation epoch. Everything else is the upstream Trainer.

Validation runs the unwrapped model, not the DDP wrapper: DDP broadcasts its
buffers in every forward, and ranks skip different numbers of clips (padding,
clips without a usable object), so wrapped forwards would deadlock. Every rank
still iterates the same number of batches, which keeps the trainer's periodic
barriers aligned; the meters are gathered once at the end.
"""

from __future__ import annotations

import gc
import json
import os

from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from sam3.train.trainer import Trainer, unwrap_ddp_if_wrapped
from sam3.train.utils.train_utils import Phase

import vdv.data  # noqa: F401  (registers the vdv dataset sources in every rank)


class _ChainedLoader:
    def __init__(self, loaders):
        self.loaders = loaders

    def __len__(self):
        return sum(len(loader) for loader in self.loaders)

    def __iter__(self):
        for loader in self.loaders:
            yield from loader


class VDVTrainer(Trainer):
    def _setup_dataloaders(self):
        self.train_dataset = None
        self.val_dataset = None
        if self.mode in ["train", "val"]:
            val_conf = self.data_conf.get(Phase.VAL, None)
            if val_conf:
                self.val_dataset = {key: instantiate(conf) for key, conf in val_conf.items()}
        if self.mode in ["train", "train_only"]:
            self.train_dataset = instantiate(self.data_conf.train)

    def run_val(self):
        if not self.val_dataset:
            return
        loaders = [ds.get_loader(epoch=int(self.epoch)) for ds in self.val_dataset.values()]
        wrapped = self.model
        self.model = unwrap_ddp_if_wrapped(wrapped)
        try:
            outs = self.val_epoch(_ChainedLoader(loaders), phase=Phase.VAL)
        finally:
            self.model = wrapped
        del loaders
        gc.collect()
        self.logger.log_dict(outs, self.epoch, axis="epoch")  # rank 0 only
        self.logger.flush()
        if self.distributed_rank == 0:
            with g_pathmgr.open(os.path.join(self.logging_conf.log_dir, "val_stats.json"), "a") as f:
                f.write(json.dumps(outs) + "\n")
