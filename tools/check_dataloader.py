"""Load a few training and evaluation clips exactly as a run would, and summarize them.

    python tools/check_dataloader.py --std ../data/ca1m/vdv [--custom DIR --ratio 0.5] [--n 4]
"""

from __future__ import annotations

import argparse
import collections
import time

import torch
from hydra.utils import instantiate

from vdv.config import RunSettings, build_config


def describe(sample) -> str:
    payload = sample.reference_payload
    appear = torch.as_tensor(payload["object_appearance"])  # (frames, objects)
    n_frames, n_obj = appear.shape[:2]
    visible = (appear > 0).float().mean().item()
    depth = torch.as_tensor(payload["depth_images"])
    depth_ok = (depth > 0).float().mean().item()
    cats = collections.Counter(q.query_text for q in sample.find_queries)
    return (f"{payload.get('scene_id', '?')}: {n_frames} frames x {n_obj} objects, "
            f"visible slots {visible:.0%}, valid depth {depth_ok:.0%}, "
            f"categories {dict(cats.most_common(5))}")


def main():
    from sam3.train.utils.train_utils import register_omegaconf_resolvers

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--std", required=True)
    p.add_argument("--custom", default=None)
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--train-split", default="train")
    p.add_argument("--test-split", default="test")
    p.add_argument("--depth-scale", type=float, default=1000.0)
    p.add_argument("--n", type=int, default=4)
    args = p.parse_args()
    register_omegaconf_resolvers()
    cfg = build_config(RunSettings(
        out="/tmp/vdv_check_dataloader", std_root=args.std, custom_root=args.custom,
        ratio=args.ratio if args.custom else 0.0, workers=1, samples_per_epoch=args.n,
        custom_train_split=args.train_split, custom_test_split=args.test_split,
        depth_scale=args.depth_scale))

    train = instantiate(cfg.trainer.data.train.dataset)
    print(f"train sources: {dict(zip(train.names, train.weights))}")
    for name, ds in train.datasets.items():
        print(f"  {name}: {len(ds.meta_file_paths)} candidate clip windows")
    t0 = time.time()
    it = iter(train)
    for i in range(args.n):
        s = next(it)
        print(f"  train[{i}] <{s.reference_payload['aggregate_source']}> {describe(s)}")
    print(f"  {(time.time() - t0) / args.n:.2f}s per training clip")

    for key, conf in (cfg.trainer.data.val or {}).items():
        ds = instantiate(conf.dataset)
        print(f"val/{key}: {ds.num_real_clips} clips")
        shown = 0
        for idx in range(min(len(ds), 50)):
            s = ds[idx]
            if s == {}:
                continue
            print(f"  {describe(s)}")
            shown += 1
            if shown == min(args.n, 2):
                break


if __name__ == "__main__":
    main()
