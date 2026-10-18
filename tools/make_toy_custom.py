"""Build the toy custom dataset: CA-1M "cushion" boxes from captures outside the standard set.

Format example and smoke-test data for scripts/train.sh. Only cushion boxes are
kept, and only frames that show a cushion. The captures are not part of
data/ca1m_10k.txt (no overlap with the standard train/test sets), and the CA-1M
standard test set has no cushions, so the custom test set is the only place
where the target category is measured.

    python tools/make_toy_custom.py --raw ../data/ca1m/raw_toy --out ../data/custom/toy
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_ca1m import convert_capture  # noqa: E402

# capture -> split
TOY_CAPTURES = {
    "ca1m-train-47204381": "train",
    "ca1m-train-42897732": "test",
    "ca1m-train-42446023": "test",
}
TOY_CATEGORIES = {"cushion"}
# '<split> <vid>' list for tools/download_ca1m.py
TOY_LIST = "\n".join(f"train {c.split('-')[-1]}" for c in TOY_CAPTURES) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw", required=True, help="folder holding the toy CA-1M captures")
    p.add_argument("--out", required=True)
    p.add_argument("--chunk", type=int, default=150)
    p.add_argument("--print-list", action="store_true", help="print the download list and exit")
    args = p.parse_args()
    if args.print_list:
        print(TOY_LIST, end="")
        return
    raw, out = os.path.abspath(args.raw), os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "scenes"), exist_ok=True)
    link = os.path.join(out, "raw")
    if os.path.islink(link):
        os.remove(link)
    os.symlink(raw, link)
    index = []
    for cap, split in TOY_CAPTURES.items():
        rows = convert_capture(os.path.join(raw, cap), raw, out, split, args.chunk, TOY_CATEGORIES)
        index.extend(rows)
        print(f"{cap}: {sum(r[1] for r in rows)} frames with {sorted(TOY_CATEGORIES)} -> {split}")
    with open(os.path.join(out, "index.json"), "w") as f:
        json.dump(index, f, indent=0)
    print(f"wrote {out}/index.json")


if __name__ == "__main__":
    main()
