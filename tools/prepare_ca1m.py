"""Convert extracted CA-1M captures into the video_data_verify dataset format.

Input: a directory with one folder per capture, as written by the CA-1M tar
extraction (scripts/download_data.sh):

    <raw>/ca1m-<split>-<vid>/<vid>/<ts>.wide/{image.png,depth.png,image/K.json,instances.json}
    <raw>/ca1m-<split>-<vid>/<vid>/<ts>.gt/RT.json
    <raw>/ca1m-<split>-<vid>/<vid>/<ts>._wide/image/size

Output (the same layout users export their own data to, see vdv/README.md):

    <out>/index.json          [[scene json, n_frames, "train"|"test"], ...]
    <out>/scenes/<scene>.json one scene per chunk of --chunk consecutive frames
    <out>/raw -> <raw>        assets are referenced through this link

CA-1M train captures become the "train" split, val captures the "test" split.
Each frame stores the official wide-camera C2W pose (gt/RT.json), the RGB
intrinsics and the per-frame 3D boxes (camera frame, meters). Depth is the
released 16-bit millimeter PNG; its intrinsics are exactly the RGB ones / 4,
so the loader's plain resize registers it to the RGB image.

    python tools/prepare_ca1m.py --raw ../data/ca1m/raw --out ../data/ca1m/vdv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _image_size(frame_dir: str) -> tuple[int, int]:
    """(width, height) of the RGB image, from the released size file."""
    size_file = frame_dir.replace(".wide", "._wide") + "/image/size"
    if os.path.exists(size_file):
        with open(size_file) as f:
            w, h = json.loads(f.read())
        return int(w), int(h)
    from PIL import Image

    with Image.open(os.path.join(frame_dir, "image.png")) as im:
        return im.size


def convert_capture(capture_dir: str, raw_root: str, out_root: str, split: str,
                    chunk: int, categories: set | None = None) -> list[list]:
    """Write the scene JSONs of one capture; return its index rows.

    With ``categories``, only boxes of those categories are kept, and only the
    frames that contain at least one of them (used by tools/make_toy_custom.py).
    """
    vid_dirs = [d for d in glob.glob(os.path.join(capture_dir, "*")) if os.path.isdir(d)]
    assert len(vid_dirs) == 1, f"expected one video folder in {capture_dir}"
    vid_dir = vid_dirs[0]
    capture = os.path.basename(capture_dir)
    stamps = sorted(os.path.basename(d)[: -len(".wide")]
                    for d in glob.glob(os.path.join(vid_dir, "*.wide")))
    assert stamps and len({len(s) for s in stamps}) == 1, f"unexpected frame names in {vid_dir}"
    # The release has a few incomplete frames (e.g. depth only); skip them.
    complete = [ts for ts in stamps if all(os.path.exists(os.path.join(vid_dir, p)) for p in (
        f"{ts}.wide/image.png", f"{ts}.wide/image/K.json", f"{ts}.wide/instances.json",
        f"{ts}.gt/RT.json"))]
    if len(complete) < len(stamps):
        print(f"{capture}: skipped {len(stamps) - len(complete)} incomplete frames")
        stamps = complete
    if categories is not None:
        stamps = [ts for ts in stamps if any(
            inst["category"] in categories
            for inst in _read_json(os.path.join(vid_dir, f"{ts}.wide", "instances.json")))]

    rows = []
    for c0 in range(0, len(stamps), chunk):
        frames, objects = {}, {}
        for ts in stamps[c0:c0 + chunk]:
            wide = os.path.join(vid_dir, f"{ts}.wide")
            rel = os.path.relpath(wide, raw_root)
            width, height = _image_size(wide)
            frames[ts] = {
                "rgb_path": f"raw/{rel}/image.png",
                "depth_path": f"raw/{rel}/depth.png",
                "rgb_intrinsic": _read_json(os.path.join(wide, "image", "K.json")),
                "camera_pose": _read_json(os.path.join(vid_dir, f"{ts}.gt", "RT.json")),
                "original_width": width,
                "original_height": height,
            }
            for inst in _read_json(os.path.join(wide, "instances.json")):
                if categories is not None and inst["category"] not in categories:
                    continue
                obj = objects.setdefault(str(inst["id"]), {"3d_category": inst["category"]})
                obj[ts] = {
                    "camera_center": inst["position"],
                    "camera_scale": inst["scale"],
                    "camera_R": inst["R"],
                    "camera_box_2d_rend": inst["box_2d_rend"],
                    "camera_box_2d_proj": inst["box_2d_proj"],
                }
        scene_id = f"{capture}_{c0 // chunk:03d}"
        scene_rel = f"scenes/{scene_id}.json"
        with open(os.path.join(out_root, scene_rel), "w") as f:
            json.dump({"scene_id": scene_id, "frames": frames, "objects": objects}, f)
        rows.append([scene_rel, len(frames), split])
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw", required=True, help="folder holding ca1m-<split>-<vid>/ captures")
    p.add_argument("--out", required=True, help="output dataset root")
    p.add_argument("--list", default=None,
                   help="optional '<split> <vid> ...' list (data/ca1m_10k.txt); default: all captures")
    p.add_argument("--chunk", type=int, default=150, help="frames per scene JSON")
    args = p.parse_args()

    raw = os.path.abspath(args.raw)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "scenes"), exist_ok=True)
    link = os.path.join(out, "raw")
    if os.path.islink(link):
        os.remove(link)
    os.symlink(raw, link)

    if args.list:
        with open(args.list) as f:
            wanted = [ln.split()[:2] for ln in f if ln.strip() and not ln.startswith("#")]
        captures = [os.path.join(raw, f"ca1m-{s}-{v}") for s, v in wanted]
    else:
        captures = sorted(glob.glob(os.path.join(raw, "ca1m-*-*")))
    missing = [c for c in captures if not os.path.isdir(c)]
    if missing:
        sys.exit(f"missing captures (run scripts/download_data.sh): {missing}")

    index = []
    for cap in captures:
        split = "train" if os.path.basename(cap).split("-")[1] == "train" else "test"
        rows = convert_capture(cap, raw, out, split, args.chunk)
        index.extend(rows)
        print(f"{os.path.basename(cap)}: {sum(r[1] for r in rows)} frames, "
              f"{len(rows)} scenes -> {split}")
    with open(os.path.join(out, "index.json"), "w") as f:
        json.dump(index, f, indent=0)
    for split in ("train", "test"):
        n = [r[1] for r in index if r[2] == split]
        print(f"{split}: {len(n)} scenes, {sum(n)} frames")
    print(f"wrote {out}/index.json")


if __name__ == "__main__":
    main()
