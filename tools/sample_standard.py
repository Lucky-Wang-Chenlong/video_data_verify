"""Sample the standard training set uniformly from the SpatialEncoder training data.

The standard data of video_data_verify is a ~10k-frame sample of HF
qicq1c/spatialencoder_full (train splits of CA-1M, ADT and Hypersim). Frames are
sampled uniformly, in windows of consecutive frames (the model trains on video
clips): a sequence is drawn with probability proportional to its frame count, a
window start uniformly inside it (no overlap with earlier windows), until the
frame budget is reached. Source proportions therefore follow the frame counts
of the full set (CA-1M 91%, ADT 6%, Hypersim 2%).

    plan    write the sampled windows to data/standard_10k_plan.json (fixed seed;
            the plan is checked in, so `build` reproduces the same data)
    build   download only the needed archives, extract the window frames, write
            one vdv dataset root per source (<out>/ca1m, <out>/adt, <out>/hypersim)
            and delete the archives

    python tools/sample_standard.py plan  --frames 10000 --window 100 --seed 0
    python tools/sample_standard.py build --plan data/standard_10k_plan.json --out ../data/standard_10k
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import random
import shutil
import sys
import tarfile
from collections import defaultdict

REPO = "qicq1c/spatialencoder_full"
SOURCES = {  # plan name -> (index json, lite_model source name)
    "CA-1M": ("BoxFromMotion/dataset/CA-1M.json", "CA-1M"),
    "ADT": ("BoxFromMotion/dataset/ADT.json", "ADT"),
    "Hypersim": ("BoxFromMotion/dataset/hyperism.json", "hyperism"),
}
# Held out of the standard set: the toy custom captures (tools/make_toy_custom.py).
EXCLUDE = {"CA-1M": {"47204381", "42897732", "42446023"}}


def _seq_id(source, path):
    name = os.path.basename(path)[: -len(".json")]
    return name


# ----------------------------------------------------------------------------- plan
def make_plan(frames, window, seed, cache):
    from huggingface_hub import HfApi, hf_hub_download

    files = set(HfApi().list_repo_files(REPO, repo_type="dataset"))
    seqs = []  # (source, seq_id, n_frames)
    for source, (index_path, _) in SOURCES.items():
        with open(hf_hub_download(REPO, index_path, repo_type="dataset", local_dir=cache)) as f:
            index = json.load(f)
        for path, n, split in index:
            if split != "train":
                continue
            sid = _seq_id(source, path)
            if sid in EXCLUDE.get(source, ()):
                continue
            if source == "CA-1M" and f"CA-1M/train/ca1m-train-{sid}.tar" not in files:
                continue
            if source == "ADT" and f"aria_digital_twin/ADT_tars/{sid}.tar" not in files:
                continue
            if source == "Hypersim" and f"hyperism_required_shards/{sid[:6]}.tar" not in files:
                continue
            seqs.append((source, sid, int(n)))
    seqs.sort()
    total = sum(n for _, _, n in seqs)
    cum, acc = [], 0
    for _, _, n in seqs:
        acc += n
        cum.append(acc)
    rng = random.Random(seed)
    taken = defaultdict(list)  # seq index -> [(start, end)]
    windows, got, tries = [], 0, 0
    while got < frames and tries < 100000:
        tries += 1
        i = bisect.bisect_right(cum, rng.randrange(total))
        source, sid, n = seqs[i]
        length = min(window, n)
        start = rng.randrange(n - length + 1)
        if any(start < e and s < start + length for s, e in taken[i]):
            continue
        taken[i].append((start, start + length))
        windows.append({"source": source, "seq": sid, "start": start, "length": length,
                        "seq_frames": n})
        got += length
    windows.sort(key=lambda w: (w["source"], w["seq"], w["start"]))
    by_source = defaultdict(lambda: [0, 0, set()])
    for w in windows:
        b = by_source[w["source"]]
        b[0] += 1
        b[1] += w["length"]
        b[2].add(w["seq"])
    summary = {s: {"windows": b[0], "frames": b[1], "sequences": len(b[2])}
               for s, b in sorted(by_source.items())}
    pool = defaultdict(int)
    for s, _, n in seqs:
        pool[s] += n
    return {"repo": REPO, "seed": seed, "window": window, "target_frames": frames,
            "pool_frames": dict(pool), "summary": summary, "windows": windows}


# ----------------------------------------------------------------------------- build
def _download(path, cache):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPO, path, repo_type="dataset", local_dir=cache)


def _subset_scene(scene, frame_ids, scene_id, rewrite=None):
    keep = set(frame_ids)
    frames = {k: dict(v) for k, v in scene["frames"].items() if k in keep}
    if rewrite:
        for fr in frames.values():
            for key in ("rgb_path", "depth_path"):
                if fr.get(key):
                    fr[key] = rewrite(fr[key])
    objects = {}
    for oid, obj in scene["objects"].items():
        entry = {k: v for k, v in obj.items() if not isinstance(v, dict) or k in keep}
        if any(isinstance(v, dict) for v in entry.values()):
            objects[oid] = entry
    return {"scene_id": scene_id, "frames": frames, "objects": objects}


def _write_scene(root, scene, rows, split="train"):
    rel = f"scenes/{scene['scene_id']}.json"
    with open(os.path.join(root, rel), "w") as f:
        json.dump(scene, f)
    rows.append([rel, len(scene["frames"]), split])


def build_ca1m(wins, root, cache):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from prepare_ca1m import convert_capture

    rows = []
    raw = os.path.join(root, "raw")
    os.makedirs(raw, exist_ok=True)
    link = os.path.join(root, "raw")  # assets live under <root>/raw, as in prepare_ca1m
    for vid, ws in sorted(_group(wins).items()):
        cap = os.path.join(raw, f"ca1m-train-{vid}")
        win_file = os.path.join(cap, "windows.json")  # window -> frame timestamps
        if not os.path.exists(win_file):
            tar_path = _download(f"CA-1M/train/ca1m-train-{vid}.tar", cache)
            with tarfile.open(tar_path) as tar:
                members = tar.getmembers()
                stamps = sorted({m.name.split("/")[1][: -len(".wide")] for m in members
                                 if m.name.count("/") >= 1 and m.name.split("/")[1].endswith(".wide")})
                per_win = {str(w["start"]): stamps[w["start"]:w["start"] + w["length"]] for w in ws}
                keep = {ts for v in per_win.values() for ts in v}
                sel = [m for m in members if m.name.count("/") >= 1
                       and m.name.split("/")[1].split(".")[0] in keep]
                tar.extractall(cap, members=sel, filter="data")
            os.remove(tar_path)
            with open(win_file, "w") as f:
                json.dump(per_win, f)
        with open(win_file) as f:
            per_win = json.load(f)
        # convert the extracted frames into one scene, then cut one scene per window
        tmp = os.path.join(root, "_tmp_scenes")
        os.makedirs(os.path.join(tmp, "scenes"), exist_ok=True)
        if not os.path.exists(os.path.join(tmp, "raw")):
            os.symlink(raw, os.path.join(tmp, "raw"))
        full_rows = convert_capture(cap, raw, tmp, "train", chunk=10 ** 9)
        with open(os.path.join(tmp, full_rows[0][0])) as f:
            full = json.load(f)
        for w in ws:
            ids = [ts for ts in per_win[str(w["start"])] if ts in full["frames"]]
            if not ids:
                continue
            _write_scene(root, _subset_scene(full, ids, f"ca1m-{vid}_{w['start']:05d}"), rows)
        shutil.rmtree(tmp)
    return rows


def build_adt(wins, root, cache):
    rows = []
    for seq, ws in sorted(_group(wins).items()):
        with open(_download(f"aria_digital_twin/train-json/{seq}.json", cache)) as f:
            scene = json.load(f)
        ordered = sorted(scene["frames"])
        ids = [ordered[i] for w in ws for i in range(w["start"], w["start"] + w["length"])]

        def rel(p):  # asset paths are "<seq>/..." under aria_digital_twin/ADT/ (tar layout too)
            return p.split("/aria_digital_twin/ADT/", 1)[1] if "/aria_digital_twin/ADT/" in p else p

        need = {os.path.normpath(rel(scene["frames"][i][k])) for i in ids
                for k in ("rgb_path", "depth_path") if scene["frames"][i].get(k)}
        done = os.path.join(root, seq, ".done")
        if not os.path.exists(done):
            tar_path = _download(f"aria_digital_twin/ADT_tars/{seq}.tar", cache)
            with tarfile.open(tar_path) as tar:
                sel = [m for m in tar.getmembers() if os.path.normpath(m.name) in need]
                if not sel:
                    names = [m.name for m in tar.getmembers()[:3]]
                    raise RuntimeError(f"ADT {seq}: no tar member matches the JSON paths "
                                       f"(tar: {names}, json: {sorted(need)[:2]})")
                tar.extractall(root, members=sel, filter="data")
            os.remove(tar_path)
            os.makedirs(os.path.dirname(done), exist_ok=True)
            open(done, "w").close()
        missing = [p for p in need if not os.path.exists(os.path.join(root, p))]
        if missing:
            raise RuntimeError(f"ADT {seq}: {len(missing)} assets missing after extraction, e.g. {missing[:2]}")
        for w in ws:
            wid = ordered[w["start"]:w["start"] + w["length"]]
            _write_scene(root, _subset_scene(scene, wid, f"adt-{seq}_{w['start']:05d}", rel), rows)
    return rows


def build_hypersim(wins, root, cache):
    rows = []
    by_vol = defaultdict(list)
    for w in wins:
        by_vol[w["seq"][:6]].append(w)
    for vol, vws in sorted(by_vol.items()):
        scenes = {}
        need = set()
        for cam, ws in _group(vws).items():
            with open(_download(f"hyperism/hyperism/train-json/{cam}.json", cache)) as f:
                scenes[cam] = json.load(f)
            ordered = sorted(scenes[cam]["frames"])
            for w in ws:
                for i in ordered[w["start"]:w["start"] + w["length"]]:
                    fr = scenes[cam]["frames"][i]
                    need.update(fr[k] for k in ("rgb_path", "depth_path") if fr.get(k))
        done = os.path.join(root, "hyperism", f".{vol}.done")
        if not os.path.exists(done):
            tar_path = _download(f"hyperism_required_shards/{vol}.tar", cache)
            with tarfile.open(tar_path) as tar:
                sel = [m for m in tar.getmembers() if m.name in need]
                tar.extractall(root, members=sel, filter="data")
            os.remove(tar_path)
            os.makedirs(os.path.dirname(done), exist_ok=True)
            open(done, "w").close()
        for cam, ws in sorted(_group(vws).items()):
            ordered = sorted(scenes[cam]["frames"])
            for w in ws:
                ids = ordered[w["start"]:w["start"] + w["length"]]
                _write_scene(root, _subset_scene(scenes[cam], ids, f"hypersim-{cam}_{w['start']:05d}"), rows)
    return rows


def _group(wins):
    g = defaultdict(list)
    for w in wins:
        g[w["seq"]].append(w)
    for ws in g.values():
        ws.sort(key=lambda w: w["start"])
    return g


BUILDERS = {"CA-1M": ("ca1m", build_ca1m), "ADT": ("adt", build_adt), "Hypersim": ("hypersim", build_hypersim)}


def build(plan_path, out, cache):
    with open(plan_path) as f:
        plan = json.load(f)
    out = os.path.abspath(out)
    cache = os.path.abspath(cache or os.path.join(out, "_download"))
    by_source = defaultdict(list)
    for w in plan["windows"]:
        by_source[w["source"]].append(w)
    for source, wins in sorted(by_source.items()):
        name, fn = BUILDERS[source]
        root = os.path.join(out, name)
        if os.path.exists(os.path.join(root, "index.json")):
            print(f"{source}: {root} exists, skipped")
            continue
        os.makedirs(os.path.join(root, "scenes"), exist_ok=True)
        rows = fn(wins, root, cache)
        with open(os.path.join(root, "index.json"), "w") as f:
            json.dump(rows, f, indent=0)
        print(f"{source}: {len(rows)} scenes, {sum(r[1] for r in rows)} frames -> {root}", flush=True)
    shutil.rmtree(cache, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--frames", type=int, default=10000)
    a.add_argument("--window", type=int, default=100)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "standard_10k_plan.json"))
    a.add_argument("--cache", default="/tmp/vdv_hf_index")
    b = sub.add_parser("build")
    b.add_argument("--plan", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--cache", default=None)
    args = p.parse_args()
    if args.cmd == "plan":
        plan = make_plan(args.frames, args.window, args.seed, args.cache)
        with open(args.out, "w") as f:
            json.dump(plan, f, indent=1)
        print(json.dumps({k: plan[k] for k in ("pool_frames", "summary")}, indent=1))
    else:
        build(args.plan, args.out, args.cache)


if __name__ == "__main__":
    main()
