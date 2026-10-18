"""Train or evaluate one video_data_verify run (one process per GPU, via torchrun).

    python -m torch.distributed.run --standalone --nproc_per_node=N -m vdv.run \
        --mode train --out OUT --std DATA/ca1m/vdv --custom MYDATA --ratio 0.2 \
        --init-ckpt CKPT [--epochs 5 --samples 1600 --lr 3e-5] [key=value ...]

Use scripts/train.sh / eval.sh, which also export the metrics afterwards.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from vdv.config import (BASE_LR, EPOCHS, SAMPLES_PER_EPOCH, SEED, VAL_STRIDE,
                        WORKERS_PER_GPU, RunSettings, build_config)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=["train", "val"], default="train")
    p.add_argument("--out", required=True)
    p.add_argument("--std", required=True, help="standard dataset root (CA-1M subset)")
    p.add_argument("--custom", default=None, help="custom dataset root")
    p.add_argument("--ratio", type=float, default=0.2, help="fraction of training clips from --custom")
    p.add_argument("--init-ckpt", default=None, help="weights to start from / evaluate")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--samples", type=int, default=SAMPLES_PER_EPOCH, help="global clips per epoch")
    p.add_argument("--workers", type=int, default=WORKERS_PER_GPU, help="dataloader workers per GPU")
    p.add_argument("--lr", type=float, default=BASE_LR)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--train-split", default="train")
    p.add_argument("--test-split", default="test")
    p.add_argument("--depth-scale", type=float, default=1000.0)
    p.add_argument("--val-sets", default="all",
                   help="comma list of test-set keys (ca1m, custom), 'standard', 'all' ('' = none)")
    p.add_argument("--val-stride", type=int, default=VAL_STRIDE)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--max-val-clips", type=int, default=None, help="per GPU (quick checks)")
    args, overrides = p.parse_known_args()
    bad = [o for o in overrides if "=" not in o]
    if bad:
        p.error(f"unknown arguments: {bad}")
    return args, overrides


def main():
    args, overrides = parse_args()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        if key not in os.environ:
            raise SystemExit("launch with python -m torch.distributed.run (see scripts/train.sh)")
    from sam3.train.final_perception import run_training
    from sam3.train.utils.train_utils import register_omegaconf_resolvers
    from omegaconf import OmegaConf

    register_omegaconf_resolvers()
    settings = RunSettings(
        out=args.out, std_root=args.std, custom_root=args.custom, ratio=args.ratio,
        mode=args.mode, init_ckpt=args.init_ckpt, epochs=args.epochs,
        samples_per_epoch=args.samples, gpus=int(os.environ["WORLD_SIZE"]),
        workers=args.workers, lr=args.lr, seed=args.seed,
        custom_train_split=args.train_split, custom_test_split=args.test_split,
        depth_scale=args.depth_scale, val_sets=[v for v in args.val_sets.split(",") if v],
        val_stride=args.val_stride, val_every=args.val_every,
        max_val_clips=args.max_val_clips, overrides=overrides)
    cfg = build_config(settings)
    if int(os.environ["RANK"]) == 0:
        out = Path(cfg.launcher.experiment_log_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    run_training(cfg)


if __name__ == "__main__":
    main()
