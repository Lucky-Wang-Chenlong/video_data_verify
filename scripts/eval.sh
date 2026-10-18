#!/usr/bin/env bash
# Evaluate a checkpoint on a custom test split and/or the standard test set (CA-1M test).
#
#   scripts/eval.sh --ckpt CKPT|released [--custom DIR] [options]
#
#   --ckpt CKPT        checkpoint path, or "released" (zero-shot baseline, A0)     [required]
#   --custom DIR       custom dataset directory; omit to evaluate the standard test sets only
#   --test-split NAME  custom test split tag in index.json                         [test]
#   --no-standard      skip the standard test set(s) (CA-1M test)
#   --gpus N           number of GPUs                                                 [8]
#   --out DIR          output directory                         [$VDV_OUT/eval_<ckpt>]
#   --depth-scale S    custom PNG depth units per meter                            [1000]
#
# Every object of a test clip (8 frames) is prompted with its GT 2D box and
# tracked; each visible (object, frame) slot is scored by the 3D IoU of its box.
# Output (--out): results/summary.md|json (overall + per category), metrics/val*.csv,
# logs/ (eval.log, log.txt), config.yaml.
set -euo pipefail
HERE="$(dirname "$(realpath "$0")")"
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

CKPT=""; CUSTOM=""; TEST=test; STD=1; GPUS=8; OUT=""; DEPTH_SCALE=1000
while [ $# -gt 0 ]; do
    case $1 in
        --ckpt) CKPT=$2; shift 2 ;;
        --custom) CUSTOM=$2; shift 2 ;;
        --test-split) TEST=$2; shift 2 ;;
        --no-standard|--no-ca1m) STD=0; shift ;;
        --gpus) GPUS=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --depth-scale) DEPTH_SCALE=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 1 ;;
    esac
done
[ -n "$CKPT" ] || { echo "--ckpt is required" >&2; usage; exit 1; }
[ -n "$CUSTOM" ] || [ "$STD" = 1 ] || { echo "nothing to evaluate" >&2; exit 1; }
[ "$CKPT" = released ] || CKPT="$(realpath "$CKPT")"
[ -z "$CUSTOM" ] || CUSTOM="$(realpath "$CUSTOM")"
[ -z "$OUT" ] || OUT="$(realpath -m "$OUT")"
source "$HERE/env.sh"
if [ "$CKPT" = released ]; then
    tag=released; CKPT="$CKPT_RELEASED"
else
    tag="$(basename "$(dirname "$(dirname "$CKPT")")")"   # run dir of .../checkpoints/x.pt
fi
OUT=${OUT:-$VDV_OUT/eval_$tag}
mkdir -p "$OUT/logs"
sets=(); [ "$STD" = 1 ] && sets+=(standard); [ -n "$CUSTOM" ] && sets+=(custom)
custom=(); [ -z "$CUSTOM" ] || custom=(--custom "$CUSTOM" --test-split "$TEST" --depth-scale "$DEPTH_SCALE")

echo "$(date +%T) eval $tag on $(IFS=,; echo "${sets[*]}"): $CKPT -> $OUT"
set +e
vdv_launch "$GPUS" --mode val --out "$OUT" --std "$VDV_STD" "${custom[@]}" \
    --init-ckpt "$CKPT" --val-sets "$(IFS=,; echo "${sets[*]}")" > "$OUT/logs/eval.log" 2>&1
rc=$?
set -e
[ "$rc" -eq 0 ] || { echo "eval failed ($rc), see $OUT/logs/eval.log" >&2; exit "$rc"; }
python "$VDV/tools/export_metrics.py" run "$OUT"
sed -n '1,12p' "$OUT/results/summary.md"
