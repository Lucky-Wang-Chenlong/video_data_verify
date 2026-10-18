"""Download HF vveicao/infinigen-traj-2k and write it in the video_data_verify format.

The dataset (2,000 synthetic Infinigen indoor videos, 73 categories) already uses
lite_model's scene-JSON format; rgb/depth paths are relative to the extracted
videos. This writes

    <out>/index.json            [["scenes/<video>.json", n_frames, "train"|"test"], ...]
                                (the dataset's val split becomes "test")
    <out>/scenes -> Infinigen/train-json   upstream scene JSONs, unchanged
    <out>/<video>/render/rgb_frames/*.jpg, <out>/<video>/depth/depth_mm_uint16/*.png

Only RGB frames and depth are extracted (mp4 previews and instance masks are
skipped); each video tar is deleted after extraction, so a re-run resumes.

    python tools/prepare_infinigen.py --out ../data/custom/infinigen [--max-videos N]
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download

REPO = "vveicao/infinigen-traj-2k"
KEEP = ("/render/rgb_frames/", "/depth/depth_mm_uint16/")


def fetch(path, cache):
    return hf_hub_download(REPO, path, repo_type="dataset", local_dir=cache)


def extract_video(video, out, cache):
    done = os.path.join(out, video, ".done")
    if os.path.exists(done):
        return video
    tar_path = fetch(f"videos/{video}.tar", cache)
    with tarfile.open(tar_path) as tar:
        members = [m for m in tar.getmembers() if any(k in m.name for k in KEEP)]
        tar.extractall(out, members=members, filter="data")
    os.remove(tar_path)
    open(done, "w").close()
    return video


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--max-videos", type=int, default=None, help="per split, for quick tests")
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()
    out = os.path.abspath(args.out)
    cache = os.path.join(out, "_download")
    os.makedirs(out, exist_ok=True)

    with open(fetch("dataset/Infinigen.json", cache)) as f:
        index = json.load(f)
    rows = []
    for split in ("train", "val"):
        items = [r for r in index if r[2] == split][: args.max_videos]
        rows += [[f"scenes/{os.path.basename(r[0])}", int(r[1]), "train" if split == "train" else "test"]
                 for r in items]
    videos = [os.path.basename(r[0])[: -len(".json")] for r in rows]

    scenes = os.path.join(out, "scenes")
    os.makedirs(scenes, exist_ok=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        for name in pool.map(lambda v: os.path.basename(fetch(f"Infinigen/train-json/{v}.json", cache)), videos):
            src = os.path.join(cache, "Infinigen", "train-json", name)
            dst = os.path.join(scenes, name)
            if not os.path.exists(dst):
                os.replace(src, dst)
    with ThreadPoolExecutor(args.jobs) as pool:
        for i, v in enumerate(pool.map(lambda v: extract_video(v, out, cache), videos), 1):
            if i % 100 == 0 or i == len(videos):
                print(f"extracted {i}/{len(videos)} videos", flush=True)

    with open(os.path.join(out, "index.json"), "w") as f:
        json.dump(rows, f, indent=0)
    for split in ("train", "test"):
        n = [r[1] for r in rows if r[2] == split]
        print(f"{split}: {len(n)} videos, {sum(n)} frames")
    print(f"wrote {out}/index.json")


if __name__ == "__main__":
    main()
