"""Check that a SpatialEncoder lite checkpoint fully covers the lite model.

Builds trainer.model from a lite_model training config (no weights), remaps the
checkpoint keys the same way sam3.model_builder._load_multiplex_tracker_checkpoint
does, and reports missing / unexpected / shape-mismatched keys per module group.

    python tools/check_lite_ckpt.py --ckpt ../ckpts/se_small_lite_wds518_e100/checkpoint.pt
"""

import argparse
import collections
import os
import sys

import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lite_model")


def remap(ckpt):
    out = {}
    for k, v in ckpt.items():
        if k.startswith("tracker.model."):
            k = k[len("tracker.model."):]
        elif k.startswith("tracker."):
            k = k[len("tracker."):]
        out.setdefault(k, v)
    return out


def group(k, depth=2):
    return ".".join(k.split(".")[:depth])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default="configs/depth/train_all_datasets_hf_wds_lite.yaml")
    p.add_argument("overrides", nargs="*", help="extra hydra overrides")
    args = p.parse_args()
    ckpt_path = os.path.abspath(args.ckpt)

    sys.path.insert(0, os.path.abspath(REPO))
    os.chdir(REPO)
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate

    with initialize_config_module("sam3.train", version_base="1.2"):
        cfg = compose(
            config_name=args.config,
            overrides=["trainer.model.use_fa3=false", "trainer.model.checkpoint_path=null",
                       *args.overrides],
        )
    model = instantiate(cfg.trainer.model, _convert_="all")
    tracker = model.tracker
    model_sd = tracker.state_dict()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    ckpt = remap(ckpt["model"] if "model" in ckpt else ckpt)

    missing = [k for k in model_sd if k not in ckpt]
    unexpected = [k for k in ckpt if k not in model_sd]
    mismatch = [(k, tuple(ckpt[k].shape), tuple(model_sd[k].shape))
                for k in model_sd if k in ckpt and tuple(ckpt[k].shape) != tuple(model_sd[k].shape)]
    n_model = sum(v.numel() for v in model_sd.values())
    n_ok = sum(model_sd[k].numel() for k in model_sd
               if k in ckpt and tuple(ckpt[k].shape) == tuple(model_sd[k].shape))

    print(f"config: {args.config}")
    print(f"model tracker tensors: {len(model_sd)}, ckpt tensors: {len(ckpt)}")
    print(f"covered: {len(model_sd) - len(missing) - len(mismatch)}/{len(model_sd)} tensors, "
          f"{n_ok / 1e6:.2f}M/{n_model / 1e6:.2f}M params ({100 * n_ok / max(n_model, 1):.2f}%)")
    for name, keys in (("missing (in model, not in ckpt)", missing),
                       ("unexpected (in ckpt, not in model)", unexpected)):
        print(f"{name}: {len(keys)}")
        for g, n in collections.Counter(group(k) for k in keys).most_common(20):
            print(f"    {g}: {n}")
    print(f"shape mismatch: {len(mismatch)}")
    for k, cs, ms in mismatch[:20]:
        print(f"    {k}: ckpt{cs} model{ms}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params under this config: {trainable / 1e6:.2f}M")
    ok = not missing and not mismatch
    print("RESULT:", "OK - checkpoint fully covers the model" if ok else "INCOMPLETE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
