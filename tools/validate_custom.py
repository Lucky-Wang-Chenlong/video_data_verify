"""Validate a video_data_verify dataset split and render its GT boxes.

    python tools/validate_custom.py --root DIR [--split train] [--depth-scale 1000] [--vis 6] [--out DIR]

Checks (ERROR = must fix before training, WARN = look at it):
  schema     index.json rows, scene JSON keys, files exist, image size
  camera     3x3 intrinsics (fx, fy > 0), 4x4 rigid camera-to-world pose
  boxes      center in front of the camera, positive size, rotation in SO(3),
             2D boxes inside the image; projected 3D box vs camera_box_2d_proj
  depth      loads, valid ratio, median in a plausible range (units / depth_scale),
             agreement with the box centers
  time       static objects keep their world position across frames
             (camera_pose @ camera_center); large drift = wrong pose direction/units
  usable     objects that pass lite_model's first-frame filter (only those are trained on)
Renders a few clips (projected 3D boxes on consecutive frames) to --out.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

ERR, WARN = "ERROR", "WARN"
FRAME_KEYS = ("rgb_path", "rgb_intrinsic")
OBJ_KEYS = ("camera_center", "camera_scale", "camera_R", "camera_box_2d_rend", "camera_box_2d_proj")


class Report:
    def __init__(self):
        self.counts = Counter()
        self.examples = defaultdict(list)

    def add(self, level, code, where):
        self.counts[(level, code)] += 1
        if len(self.examples[(level, code)]) < 3:
            self.examples[(level, code)].append(where)

    def errors(self):
        return sum(n for (lvl, _), n in self.counts.items() if lvl == ERR)

    def print(self):
        for (lvl, code), n in sorted(self.counts.items()):
            print(f"  {lvl:5s} {code}: {n}   e.g. {'; '.join(self.examples[(lvl, code)])}")


def corners(center, scale, rot):
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float64)
    return np.asarray(center) + (signs * np.asarray(scale) / 2) @ np.asarray(rot).T


def project(K, pts):
    pts = np.asarray(pts, np.float64)
    uv = (K @ pts.T).T
    return uv[:, :2] / uv[:, 2:3]


def iou2d(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_depth(path, scale):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(np.float32)
    if ext == ".npz":
        with np.load(path) as z:
            return (z["depth"] if "depth" in z.files else z[z.files[0]]).astype(np.float32)
    from PIL import Image

    return np.asarray(Image.open(path)).astype(np.float32) / scale


def check_scene(root, rel, n_frames, split, args, rep, stats, usable_filter):
    path = os.path.join(root, rel)
    if not os.path.exists(path):
        rep.add(ERR, "scene json missing", rel)
        return None
    with open(path) as f:
        scene = json.load(f)
    for key in ("scene_id", "frames", "objects"):
        if key not in scene:
            rep.add(ERR, f"scene missing '{key}'", rel)
            return None
    frames = scene["frames"]
    if len(frames) != n_frames:
        rep.add(WARN, "index n_frames != len(frames)", f"{rel}: {n_frames} vs {len(frames)}")
    ids = sorted(frames)
    stats["frames"] += len(ids)
    stats["scenes"] += 1
    sizes = {}
    for fid in ids:
        fr = frames[fid]
        where = f"{rel}#{fid}"
        miss = [k for k in FRAME_KEYS if k not in fr]
        if miss:
            rep.add(ERR, f"frame missing {miss}", where)
            continue
        K = np.asarray(fr["rgb_intrinsic"], np.float64)
        if K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0:
            rep.add(ERR, "bad rgb_intrinsic", where)
        pose = np.asarray(fr.get("camera_pose", np.eye(4)), np.float64)
        if "camera_pose" not in fr:
            rep.add(WARN, "no camera_pose (identity used; temporal checks skipped)", where)
        elif pose.shape != (4, 4) or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5) or \
                not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-3):
            rep.add(ERR, "camera_pose not a rigid 4x4 camera-to-world matrix", where)
        rgb = os.path.join(root, fr["rgb_path"])
        if not os.path.exists(rgb):
            rep.add(ERR, "rgb file missing", where)
            continue
        if "original_width" not in fr or "original_height" not in fr:
            rep.add(WARN, "no original_width/height (filter assumes 1024x1024)", where)
        sizes[fid] = (fr.get("original_width"), fr.get("original_height"))
        if fr.get("depth_path"):
            dpath = os.path.join(root, fr["depth_path"])
            if not os.path.exists(dpath):
                rep.add(ERR, "depth file missing", where)
        else:
            stats["frames_without_depth"] += 1
    # image size check on a few frames
    from PIL import Image

    for fid in ids[:: max(1, len(ids) // 3)][:3]:
        fr = frames[fid]
        rgb = os.path.join(root, fr["rgb_path"])
        if not os.path.exists(rgb):
            continue
        with Image.open(rgb) as im:
            w, h = im.size
        if sizes.get(fid, (w, h)) != (w, h) and None not in sizes.get(fid, (None,)):
            rep.add(ERR, "original_width/height != image size", f"{rel}#{fid}: {sizes[fid]} vs {(w, h)}")
        sizes[fid] = (w, h)
        if fr.get("depth_path") and os.path.exists(os.path.join(root, fr["depth_path"])):
            d = load_depth(os.path.join(root, fr["depth_path"]), args.depth_scale)
            valid = d[np.isfinite(d) & (d > 0)]
            stats["depth_valid"].append(valid.size / max(d.size, 1))
            if valid.size == 0:
                rep.add(WARN, "depth map has no valid pixel", f"{rel}#{fid}")
                continue
            med = float(np.median(valid))
            stats["depth_median"].append(med)
            if not 0.1 <= med <= 50:
                rep.add(ERR, "depth median outside 0.1-50 m (check --depth-scale)", f"{rel}#{fid}: {med:.2f} m")
            # box centers vs depth
            K = np.asarray(fr["rgb_intrinsic"], np.float64)
            sy, sx = d.shape[0] / h, d.shape[1] / w
            for oid, obj in scene["objects"].items():
                o = obj.get(fid)
                if not o or "camera_center" not in o:
                    continue
                c = np.asarray(o["camera_center"], np.float64)
                if c[2] <= 0.1:
                    continue
                u, v = project(K, c[None])[0]
                if 0 <= u < w and 0 <= v < h:
                    z = d[int(v * sy), int(u * sx)]
                    if z > 0:
                        stats["depth_vs_center"].append(float(z - c[2]) / max(float(np.max(o["camera_scale"])), 0.1))

    # World "up" from the cameras: OpenCV camera -Y points roughly up for upright views
    # (cameras often pitch down, so snap the estimate to the nearest world axis).
    poses = [np.asarray(frames[f]["camera_pose"], np.float64) for f in ids
             if "camera_pose" in frames[f] and np.isfinite(np.asarray(frames[f]["camera_pose"], np.float64)).all()]
    up = None
    if poses:
        est = -np.mean([p[:3, 1] for p in poses], axis=0)
        if np.linalg.norm(est) > 0.3:
            ax = int(np.argmax(np.abs(est)))
            up = np.zeros(3)
            up[ax] = np.sign(est[ax])

    world = defaultdict(list)
    for oid, obj in scene["objects"].items():
        cat = obj.get("3d_category") or obj.get("category_name")
        if cat is None:
            rep.add(ERR, "object without 3d_category", f"{rel}:{oid}")
        stats["objects"] += 1
        stats["categories"][cat] += 1
        for fid, o in obj.items():
            if not isinstance(o, dict):
                continue
            where = f"{rel}#{fid}:{oid}"
            if fid not in frames:
                rep.add(ERR, "object frame id not in frames", where)
                continue
            miss = [k for k in OBJ_KEYS if k not in o]
            if miss:
                rep.add(ERR, f"object entry missing {miss}", where)
                continue
            stats["boxes"] += 1
            c = np.asarray(o["camera_center"], np.float64)
            s = np.asarray(o["camera_scale"], np.float64)
            R = np.asarray(o["camera_R"], np.float64)
            if s.shape != (3,) or np.any(s <= 0):
                rep.add(ERR, "camera_scale must be 3 positive sizes (m)", where)
                continue
            if R.shape != (3, 3) or not np.allclose(R.T @ R, np.eye(3), atol=1e-2) or np.linalg.det(R) < 0.9:
                rep.add(ERR, "camera_R not a rotation matrix", where)
                continue
            if c[2] <= 0:
                rep.add(WARN, "box center behind the camera", where)
                continue
            if np.max(s) > 10:
                rep.add(WARN, "box larger than 10 m (units?)", where)
            fr = frames[fid]
            K = np.asarray(fr["rgb_intrinsic"], np.float64)
            w, h = sizes.get(fid, (None, None))
            rend = np.asarray(o["camera_box_2d_rend"], np.float64)
            if rend.shape != (4,) or rend[2] <= rend[0] or rend[3] <= rend[1]:
                rep.add(ERR, "camera_box_2d_rend must be x1,y1,x2,y2 with x2>x1, y2>y1", where)
            elif w and (rend[0] < -1 or rend[1] < -1 or rend[2] > w + 1 or rend[3] > h + 1):
                rep.add(WARN, "camera_box_2d_rend outside the image (should be the visible box)", where)
            cs = corners(c, s, R)
            if np.all(cs[:, 2] > 0.05):
                uv = project(K, cs)
                proj = [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]
                stats["proj_iou"].append(iou2d(proj, list(o["camera_box_2d_proj"])))
            if "camera_pose" in fr:
                pose = np.asarray(fr["camera_pose"], np.float64)
                world[oid].append(pose[:3, :3] @ c + pose[:3, 3])
                if up is not None:
                    v = (pose[:3, :3] @ R).T @ up  # up in box coordinates
                    ax = int(np.argmax(np.abs(v)))
                    stats["vertical_axis"][f"{'XYZ'[ax]}{'-' if v[ax] > 0 else '+'}"] += 1
    for oid, pts in world.items():
        if len(pts) >= 2:
            pts = np.asarray(pts)
            stats["world_drift"].append(float(np.linalg.norm(pts - pts.mean(0), axis=1).max()))

    if usable_filter is not None and ids:
        first = ids[0]
        fw, fh = sizes.get(first, (1024, 1024))
        objs = {k: v for k, v in scene["objects"].items() if v.get(first) is not None}
        try:
            kept = usable_filter(objs, first, resolution=(fh or 1024, fw or 1024))
            stats["usable_first_frame"].append((len(kept), len(objs)))
        except Exception as e:  # noqa: BLE001
            rep.add(WARN, f"lite_model filter failed: {type(e).__name__}", rel)
    return scene


def render(root, scene, out_dir, n_frames=4, max_objs=12):
    import cv2

    frames = scene["frames"]
    ids = sorted(frames)
    counts = [sum(1 for o in scene["objects"].values() if fid in o) for fid in ids]
    start = int(np.argmax(counts))
    start = max(0, min(start, len(ids) - n_frames))
    tiles = []
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    rng = random.Random(0)
    colors = {}
    for fid in ids[start:start + n_frames]:
        fr = frames[fid]
        img = cv2.imread(os.path.join(root, fr["rgb_path"]))
        if img is None:
            continue
        K = np.asarray(fr["rgb_intrinsic"], np.float64)
        for oid, obj in list(scene["objects"].items())[:200]:
            o = obj.get(fid)
            if not isinstance(o, dict) or "camera_center" not in o:
                continue
            if oid not in colors:
                if len(colors) >= max_objs:
                    continue
                colors[oid] = tuple(int(x) for x in rng.sample(range(64, 256), 3))
            cs = corners(o["camera_center"], o["camera_scale"], o["camera_R"])
            if np.any(cs[:, 2] <= 0.05):
                continue
            uv = project(K, cs).astype(int)
            for a, b in edges:
                cv2.line(img, tuple(uv[a]), tuple(uv[b]), colors[oid], 2, cv2.LINE_AA)
            x1, y1 = (int(v) for v in o["camera_box_2d_rend"][:2])
            label = obj.get("3d_category") or obj.get("category_name") or "?"
            cv2.putText(img, f"{label}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        colors[oid], 1, cv2.LINE_AA)
        cv2.putText(img, f"{scene['scene_id']} {fid}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        scale = 480 / img.shape[0]
        tiles.append(cv2.resize(img, (int(img.shape[1] * scale), 480)))
    if tiles:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{scene['scene_id']}.jpg")
        cv2.imwrite(path, np.concatenate(tiles, axis=1))
        return path
    return None


def _pct(values, q):
    return float(np.percentile(values, q)) if values else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", required=True)
    p.add_argument("--split", default=None, help="split tag in index.json (default: every split)")
    p.add_argument("--depth-scale", type=float, default=1000.0)
    p.add_argument("--vis", type=int, default=6, help="scenes to render per split")
    p.add_argument("--out", default=None, help="render directory [<root>/_vis]")
    args = p.parse_args()
    root = os.path.abspath(args.root)
    index_path = os.path.join(root, "index.json")
    if not os.path.exists(index_path):
        sys.exit(f"ERROR: {index_path} not found")
    with open(index_path) as f:
        index = json.load(f)
    try:
        from sam3.train.data.ca1m_dataset import _filter_object
    except Exception:  # noqa: BLE001
        _filter_object = None
    splits = sorted({row[2] for row in index}) if args.split is None else [args.split]
    total_errors = 0
    for split in splits:
        rows = [r for r in index if len(r) == 3 and r[2] == split]
        rep = Report()
        stats = defaultdict(list)
        for key in ("frames", "scenes", "objects", "boxes", "frames_without_depth"):
            stats[key] = 0
        stats["categories"] = Counter()
        stats["vertical_axis"] = Counter()  # which box axis points down ("Y+" = lite_model convention)
        bad_rows = [r for r in index if len(r) != 3]
        for r in bad_rows[:3]:
            rep.add(ERR, "index row must be [scene_json, n_frames, split]", str(r))
        scenes = []
        for rel, n, _ in rows:
            scene = check_scene(root, rel, int(n), split, args, rep, stats, _filter_object)
            if scene is not None:
                scenes.append(scene)
        print(f"\n== split '{split}': {stats['scenes']} scenes, {stats['frames']} frames, "
              f"{stats['objects']} objects, {stats['boxes']} boxes")
        if not rows:
            rep.add(ERR, "no index rows for this split", split)
        print("  categories:", dict(stats["categories"].most_common(15)),
              "..." if len(stats["categories"]) > 15 else "")
        if stats["proj_iou"]:
            low = np.mean(np.asarray(stats["proj_iou"]) < 0.5)
            print(f"  projected 3D box vs camera_box_2d_proj: median IoU {np.median(stats['proj_iou']):.3f}, "
                  f"{100 * low:.1f}% below 0.5")
            if low > 0.1:
                rep.add(ERR, "projected 3D boxes disagree with camera_box_2d_proj (axes/units/intrinsics?)",
                        f"{100 * low:.0f}% of boxes")
        if stats["depth_median"]:
            print(f"  depth: median {np.median(stats['depth_median']):.2f} m, valid "
                  f"{100 * np.mean(stats['depth_valid']):.0f}% of pixels; "
                  f"{stats['frames_without_depth']} frames without depth")
        if stats["depth_vs_center"]:
            dv = np.abs(stats["depth_vs_center"])
            far = float(np.mean(dv > 2))
            print(f"  depth at box center - box z: median |diff| {np.median(dv):.2f} box sizes, "
                  f"{100 * far:.0f}% beyond 2 box sizes")
            if far > 0.3:
                rep.add(WARN, "depth and box centers disagree (depth units or registration?)",
                        f"{100 * far:.0f}% of sampled boxes")
        if stats["world_drift"]:
            drift = stats["world_drift"]
            print(f"  world-position drift per object: median {np.median(drift):.3f} m, "
                  f"p90 {_pct(drift, 90):.3f} m")
            if np.median(drift) > 0.2:
                rep.add(WARN, "static objects move in world space (pose is world-to-camera? units?)",
                        f"median drift {np.median(drift):.2f} m")
        if stats["vertical_axis"]:
            va = stats["vertical_axis"]
            frac = va["Y+"] / sum(va.values())
            print(f"  box axis pointing down (lite_model expects local +Y): {dict(va.most_common(3))}")
            if frac < 0.5:
                rep.add(WARN, "box axes: local +Y should point down (gravity); yaw and size targets "
                              "assume it, e.g. Z-up boxes need camera_R' = [X, -Z, Y], scale' = [sx, sz, sy]",
                        f"only {100 * frac:.0f}% of boxes have +Y down")
        if stats["usable_first_frame"]:
            kept = sum(k for k, _ in stats["usable_first_frame"])
            seen = sum(n for _, n in stats["usable_first_frame"])
            print(f"  usable objects (lite_model filter, scene first frames): {kept}/{seen}")
            if seen and kept == 0:
                rep.add(ERR, "no object passes the lite_model filter (size >= 0.1 m / 30 px, z > 0.1)", split)
        rep.print()
        total_errors += rep.errors()
        out_dir = args.out or os.path.join(root, "_vis")
        rng = random.Random(0)
        for scene in rng.sample(scenes, min(args.vis, len(scenes))):
            path = render(root, scene, os.path.join(out_dir, split))
            if path:
                print(f"  rendered {path}")
    print(f"\n{'OK' if total_errors == 0 else f'{total_errors} ERRORs'}")
    sys.exit(1 if total_errors else 0)


if __name__ == "__main__":
    main()
