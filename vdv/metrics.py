"""3D box quality meter for GT-2D-box-prompted video evaluation.

The model is prompted with each object's GT 2D box and tracks it through the
clip, so every (object, frame) slot has exactly one prediction; there is no
detection/recall step and no AP. Per slot with a visible GT object the loss
records the 3D IoU and the translation / depth / yaw / scale errors of the
selected box candidate (``loss.meter_container``). This meter collects them
together with the slot's category (the query text) and reports, overall and
per category:

    miou       mean 3D IoU
    acc25      fraction of slots with 3D IoU >= 0.25  (likewise acc50, acc10)
    trans_err  center error [m];  depth_err [m];  scale_err (1 - volume IoU)
    yaw_err    orientation error about the vertical [deg], folded to [0, 45] by the
               cuboid's 90-degree footprint symmetry; yaw_err_raw = lite_model's unfolded
               |yaw difference| in [0, 180], which depends on each dataset's horizontal
               axis order (about 90 for PCA-fitted boxes even when they are right)
    n          number of evaluated slots

Errors are in meters: lite_model normalizes distances by ``norm_scale``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict

import numpy as np

from sam3.train.metric import BaseMetric, _to_list, gather_values, is_dist_avail_and_initialized

_FIELDS = ("ious", "translation_errors", "depth_errors", "yaw_errors", "scale_errors")


def _categories(batch, n: int) -> list:
    """Category of each metric slot, in the loss's stage-major order."""
    if batch is None:
        return ["?"] * n
    texts = batch.find_text_batch
    cats = []
    for stage in batch.find_inputs:
        cats.extend(texts[int(t)] for t in _to_list(stage.text_ids))
    return cats if len(cats) == n else ["?"] * n


def summarize(rows) -> Dict[str, float]:
    """rows: (category, iou, trans, depth, yaw_rad, scale). Overall + per-category stats."""
    out = {}
    groups = defaultdict(list)
    for row in rows:
        groups[row[0]].append(row[1:])
    for name, items in [("", [r[1:] for r in rows])] + sorted(groups.items()):
        if not items:
            continue
        arr = np.asarray(items, dtype=np.float64)
        prefix = f"cat/{name}/" if name else ""
        iou = np.nan_to_num(arr[:, 0], nan=0.0)
        out[prefix + "n"] = float(len(arr))
        out[prefix + "miou"] = float(iou.mean())
        out[prefix + "acc25"] = float((iou >= 0.25).mean())
        out[prefix + "acc50"] = float((iou >= 0.5).mean())
        if not name:
            out["acc10"] = float((iou >= 0.10).mean())
            for col, key, scale in ((1, "trans_err", 1.0), (2, "depth_err", 1.0),
                                    (3, "yaw_err_raw", 180.0 / math.pi), (4, "scale_err", 1.0)):
                vals = arr[:, col][np.isfinite(arr[:, col])]
                out[key] = float(vals.mean() * scale) if len(vals) else float("nan")
            # A cuboid's footprint is unchanged by a 90 deg turn with width/length
            # swapped, so fold the yaw error into [0, 45] deg: comparable across
            # datasets whose horizontal axis order differs (e.g. PCA-fitted boxes).
            yaw = arr[:, 3][np.isfinite(arr[:, 3])] % (math.pi / 2)
            yaw = np.minimum(yaw, math.pi / 2 - yaw)
            out["yaw_err"] = float(np.degrees(yaw).mean()) if len(yaw) else float("nan")
    return out


class Box3DMeter(BaseMetric):
    def __init__(self, norm_scale: float = 2.5):
        self.norm_scale = float(norm_scale)
        super().__init__()

    def reset(self):
        self.rows = []

    def update(self, find_stages, find_metadatas, batch=None, **kwargs):
        exist = _to_list(find_stages.get("gt_existence"))
        if not exist:
            return
        n = len(exist)
        cols = []
        for field in _FIELDS:
            vals = _to_list(find_stages.get(field))
            cols.append(vals if len(vals) == n else [float("nan")] * n)
        cats = _categories(batch, n)
        for i in range(n):
            if int(exist[i]) != 1:
                continue
            self.rows.append((
                cats[i], float(cols[0][i]),
                float(cols[1][i]) * self.norm_scale, float(cols[2][i]) * self.norm_scale,
                float(cols[3][i]), float(cols[4][i]),
            ))

    def compute_local(self) -> Dict[str, float]:
        return summarize(self.rows)

    def compute_synced(self) -> Dict[str, float]:
        if not is_dist_avail_and_initialized():
            return self.compute_local()
        import torch.distributed as dist

        rows = gather_values([self.rows])
        if dist.get_rank() != 0:
            return {}
        return summarize([r for chunk in rows for r in chunk])
