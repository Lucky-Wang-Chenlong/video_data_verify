"""Export the metrics of video_data_verify runs.

    python tools/export_metrics.py run OUT                 # one train.sh / eval.sh run
    python tools/export_metrics.py compare RUN... [--to compare.md]

``run`` writes into OUT:
    metrics/train.csv      every 10 steps: total loss, loss terms, learning rate (from tensorboard)
    metrics/val.csv        one row per validation x test set: miou, acc10/25/50, errors, n
    metrics/val_per_category.csv   same, per category
    results/summary.json   metrics of the last validation (= final weights)
    results/summary.md     the same as tables (overall per test set + per category)

Metrics (vdv/metrics.py): every object of a test clip is prompted with its GT
2D box in the first frame and tracked through the 8-frame clip; each visible
(object, frame) slot is scored by the 3D IoU of its predicted box.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

OVERALL = ["miou", "acc10", "acc25", "acc50", "trans_err", "depth_err", "yaw_err", "yaw_err_raw", "scale_err", "n"]
PER_CAT = ["miou", "acc25", "acc50", "n"]
HEADLINE = ["miou", "acc25", "acc50"]
_KEY = re.compile(r"^Meters_train/val_(?P<set>[^/]+)/box3d/(?P<rest>.+)$")


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_val(stats: dict) -> dict:
    """{set: {"overall": {metric: v}, "cat": {name: {metric: v}}, "epoch": e}}."""
    out = {}
    for key, value in stats.items():
        m = _KEY.match(key)
        if not m:
            continue
        entry = out.setdefault(m["set"], {"overall": {}, "cat": defaultdict(dict),
                                          "epoch": stats.get("Trainer/epoch")})
        rest = m["rest"]
        if rest.startswith("cat/"):
            name, metric = rest[4:].rsplit("/", 1)
            entry["cat"][name][metric] = value
        else:
            entry["overall"][rest] = value
    for entry in out.values():
        entry["cat"] = dict(entry["cat"])
    return out


def _write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _fmt(v, metric):
    if v is None:
        return "–"
    if metric == "n":
        return f"{int(v)}"
    if metric in ("miou", "acc10", "acc25", "acc50"):
        return f"{100 * v:.1f}"
    return f"{v:.1f}" if metric.startswith("yaw_err") else f"{v:.3f}"


def export_train_curve(out):
    events = sorted(glob.glob(os.path.join(out, "tensorboard", "events.out.tfevents.*")))
    if not events:
        return 0
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.util import tensor_util

    rows = defaultdict(dict)
    for path in events:
        for ev in EventFileLoader(path).Load():
            for v in ev.summary.value:
                tag = v.tag
                if tag.startswith("Step_Losses/Losses/train_all_loss"):
                    name = "loss" if tag.endswith("train_all_loss") else tag.rsplit("/", 1)[-1]
                elif re.fullmatch(r"Optim/\d+_/lr", tag):
                    name = "lr_" + tag.split("/")[1].rstrip("_")
                else:
                    continue
                value = (float(tensor_util.make_ndarray(v.tensor)) if v.HasField("tensor")
                         else float(v.simple_value))
                rows[int(ev.step)][name] = value
    if not rows:
        return 0
    lrs = sorted({k for r in rows.values() for k in r if k.startswith("lr_")})
    terms = sorted({k for r in rows.values() for k in r if k != "loss" and not k.startswith("lr_")})
    # The head learning rate is the largest parameter-group LR.
    header = ["step", "loss", "lr", *terms]
    # LR is logged at its own steps: use the latest value at or before each loss
    # row (the first logged value for rows before it).
    lr_steps = [(s, max(rows[s][k] for k in lrs if k in rows[s]))
                for s in sorted(rows) if any(k in rows[s] for k in lrs)]
    table = []
    for step in sorted(s for s in rows if "loss" in rows[s]):
        r = rows[step]
        earlier = [v for s, v in lr_steps if s <= step]
        lr = earlier[-1] if earlier else (lr_steps[0][1] if lr_steps else None)
        table.append([step, r["loss"], lr, *[r.get(t) for t in terms]])
    _write_csv(os.path.join(out, "metrics", "train.csv"), header, table)
    return len(table)


def export_run(out):
    out = os.path.abspath(out)
    n_train = export_train_curve(out)
    vals = [parse_val(s) for s in _read_jsonl(os.path.join(out, "logs", "val_stats.json"))]
    vals = [v for v in vals if v]
    rows, cat_rows = [], []
    for i, val in enumerate(vals):
        for name, entry in sorted(val.items()):
            epoch = entry["epoch"] if entry["epoch"] is not None else i
            rows.append([i, epoch, name, *[entry["overall"].get(m) for m in OVERALL]])
            for cat, mets in sorted(entry["cat"].items()):
                cat_rows.append([i, epoch, name, cat, *[mets.get(m) for m in PER_CAT]])
    if rows:
        _write_csv(os.path.join(out, "metrics", "val.csv"),
                   ["val_round", "epoch", "test_set", *OVERALL], rows)
        _write_csv(os.path.join(out, "metrics", "val_per_category.csv"),
                   ["val_round", "epoch", "test_set", "category", *PER_CAT], cat_rows)
    print(f"{out}: {n_train} train rows, {len(vals)} validation rounds")
    if not vals:
        return None
    final = vals[-1]
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    with open(os.path.join(out, "results", "summary.json"), "w") as f:
        json.dump(final, f, indent=1)
    with open(os.path.join(out, "results", "summary.md"), "w") as f:
        f.write(summary_md(os.path.basename(out), final))
    return final


def summary_md(name, final, max_cats=30):
    lines = [f"# {name}", "",
             "3D IoU of GT-2D-box-prompted, tracked boxes (8-frame clips). "
             "miou / acc = %, errors in m, yaw in deg (yaw_err folded to 0-45 by box symmetry, yaw_err_raw unfolded).", "",
             "| test set | " + " | ".join(OVERALL) + " |",
             "|---|" + "---|" * len(OVERALL)]
    for s, e in sorted(final.items()):
        lines.append(f"| {s} | " + " | ".join(_fmt(e["overall"].get(m), m) for m in OVERALL) + " |")
    for s, e in sorted(final.items()):
        cats = sorted(e["cat"].items(), key=lambda kv: -kv[1].get("n", 0))
        lines += ["", f"## {s}: per category" + (f" (top {max_cats} by n)" if len(cats) > max_cats else ""),
                  "", "| category | " + " | ".join(PER_CAT) + " |", "|---|" + "---|" * len(PER_CAT)]
        for c, mets in cats[:max_cats]:
            lines.append(f"| {c} | " + " | ".join(_fmt(mets.get(m), m) for m in PER_CAT) + " |")
    return "\n".join(lines) + "\n"


def load_final(run):
    path = os.path.join(run, "results", "summary.json")
    if not os.path.exists(path):
        export_run(run)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compare(runs, to=None):
    finals = {os.path.basename(os.path.abspath(r)): load_final(r) for r in runs}
    sets = sorted({s for f in finals.values() if f for s in f})
    # custom first: it is the target, ca1m is the forgetting check
    sets = sorted(sets, key=lambda s: (s != "custom", s))
    head = ["run"] + [f"{s} {m}" for s in sets for m in HEADLINE]
    lines = ["| " + " | ".join(head) + " |", "|---|" + "---|" * (len(head) - 1)]
    for name, final in finals.items():
        cells = [name]
        for s in sets:
            e = (final or {}).get(s, {}).get("overall", {})
            cells += [_fmt(e.get(m), m) for m in HEADLINE]
        lines.append("| " + " | ".join(cells) + " |")
    if "custom" in sets:
        cats = sorted({c for f in finals.values() if f and "custom" in f for c in f["custom"]["cat"]})
        lines += ["", "custom test set, mIoU (%) per category:", "",
                  "| run | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
        for name, final in finals.items():
            cat = (final or {}).get("custom", {}).get("cat", {})
            lines.append(f"| {name} | " + " | ".join(_fmt(cat.get(c, {}).get("miou"), "miou")
                                                    for c in cats) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    if to:
        with open(to, "w") as f:
            f.write(text)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out")
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+")
    c.add_argument("--to", default=None)
    args = p.parse_args()
    if args.cmd == "run":
        export_run(args.out)
    else:
        compare(args.runs, args.to)


if __name__ == "__main__":
    main()
