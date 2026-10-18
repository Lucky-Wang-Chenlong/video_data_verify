#!/usr/bin/env bash
# Check a custom video dataset (schema, cameras, boxes, depth, temporal
# consistency) and render GT boxes on a few clips.
#
#   scripts/validate.sh --custom DIR [--split NAME] [--vis N] [--out DIR] [--depth-scale S]
#
#   --custom DIR     custom dataset directory (see vdv/README.md)       [required]
#   --split NAME     split tag in index.json                       [every split]
#   --vis N          scenes to render per split                               [6]
#   --out DIR        render directory              [$VDV_OUT/vis/<dataset>/<split>]
#   --depth-scale S  PNG depth units per meter (1000 = millimeters)       [1000]
#
# Exit code 1 if there are ERRORs. Fix all of them before training.
set -euo pipefail
HERE="$(dirname "$(realpath "$0")")"
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

CUSTOM=""; SPLIT=(); NVIS=6; OUT=""; DEPTH_SCALE=1000
while [ $# -gt 0 ]; do
    case $1 in
        --custom) CUSTOM=$2; shift 2 ;;
        --split) SPLIT=(--split "$2"); shift 2 ;;
        --vis) NVIS=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --depth-scale) DEPTH_SCALE=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 1 ;;
    esac
done
[ -n "$CUSTOM" ] || { echo "--custom DIR is required" >&2; usage; exit 1; }
CUSTOM="$(realpath "$CUSTOM")"
source "$HERE/env.sh"
OUT="$(realpath -m "${OUT:-$VDV_OUT/vis/$(basename "$CUSTOM")}")"
python "$VDV/tools/validate_custom.py" --root "$CUSTOM" "${SPLIT[@]}" \
    --depth-scale "$DEPTH_SCALE" --vis "$NVIS" --out "$OUT"
