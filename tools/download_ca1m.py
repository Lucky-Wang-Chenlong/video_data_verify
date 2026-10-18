"""Download and extract CA-1M captures listed in a '<split> <video_id> ...' file.

CA-1M (apple/ml-cubifyanything) is released under CC-BY-NC-ND 4.0; pass
--accept-license to confirm you accept it. Captures already extracted under
--raw are skipped, so an interrupted run can simply be restarted.

    python tools/download_ca1m.py --list data/ca1m_10k.txt --raw ../data/ca1m/raw --accept-license
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://ml-site.cdn-apple.com/datasets/ca1m/{split}/ca1m-{split}-{vid}.tar"


def done(capture_dir: str) -> bool:
    return bool(glob.glob(os.path.join(capture_dir, "*", "world.gt", "instances.json")))


def fetch(split: str, vid: str, raw: str) -> str:
    name = f"ca1m-{split}-{vid}"
    dst = os.path.join(raw, name)
    if done(dst):
        return f"{name}: already present"
    tmp_tar = os.path.join(raw, f".{name}.tar.part")
    tmp_dir = os.path.join(raw, f".{name}.extract")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    urllib.request.urlretrieve(URL.format(split=split, vid=vid), tmp_tar)
    with tarfile.open(tmp_tar) as tar:
        tar.extractall(tmp_dir, filter="data")
    os.remove(tmp_tar)
    if not done(tmp_dir):
        raise RuntimeError(f"{name}: unexpected archive layout")
    shutil.rmtree(dst, ignore_errors=True)
    os.rename(tmp_dir, dst)
    n = len(glob.glob(os.path.join(dst, "*", "*.wide")))
    return f"{name}: {n} frames"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--list", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--accept-license", action="store_true",
                   help="accept the CA-1M CC-BY-NC-ND 4.0 license")
    args = p.parse_args()
    if not args.accept_license:
        sys.exit("CA-1M is released under CC-BY-NC-ND 4.0 "
                 "(https://creativecommons.org/licenses/by-nc-nd/4.0/); "
                 "re-run with --accept-license to accept it.")
    os.makedirs(args.raw, exist_ok=True)
    with open(args.list) as f:
        items = [ln.split()[:2] for ln in f if ln.strip() and not ln.startswith("#")]
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(fetch, s, v, args.raw) for s, v in items]
        failed = 0
        for fut in futures:
            try:
                print(fut.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAILED: {e}", flush=True)
    if failed:
        sys.exit(f"{failed} captures failed; re-run to retry")


if __name__ == "__main__":
    main()
