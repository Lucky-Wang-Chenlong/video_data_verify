#!/usr/bin/env bash
# Full comparison on one custom dataset: zero-shot baseline + one finetune per
# custom-data ratio, then one table.
#
#   scripts/compare.sh --custom DIR [--ratios "0 0.2"] [--out DIR] [train.sh options]
#
#   --ratios "R..."  custom-data ratios to train; 0 is the control       ["0 0.2"]
#   --out DIR        parent of all runs                [$VDV_OUT/compare_<dataset>]
#   other options (--gpus, --epochs, --samples, --lr, --*-split, key=value, ...) go to train.sh
#
# Output (--out):
#   zeroshot/      released checkpoint (A0)
#   ratio<R>/      one train.sh run per ratio (A1 = ratio0, A2 = ratio0.2, ...)
#   compare.md     mIoU / acc25 / acc50 per run x test set, and custom mIoU per category
set -euo pipefail
HERE="$(dirname "$(realpath "$0")")"
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

CUSTOM=""; RATIOS="0 0.2"; OUT=""; eval_opt=(); pass=()
while [ $# -gt 0 ]; do
    case $1 in
        --custom) CUSTOM=$2; shift 2 ;;
        --ratios) RATIOS=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --gpus|--test-split|--depth-scale) eval_opt+=("$1" "$2"); pass+=("$1" "$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *=*) pass+=("$1"); shift ;;
        *) pass+=("$1" "$2"); shift 2 ;;
    esac
done
[ -n "$CUSTOM" ] || { echo "--custom DIR is required" >&2; usage; exit 1; }
CUSTOM="$(realpath "$CUSTOM")"
OUT="$(realpath -m "${OUT:-$(bash -c "source '$HERE/env.sh'; echo \$VDV_OUT")/compare_$(basename "$CUSTOM")}")"
mkdir -p "$OUT"

runs=("$OUT/zeroshot")
if [ -f "$OUT/zeroshot/results/summary.json" ]; then
    echo "skip $OUT/zeroshot (done)"
else
    bash "$HERE/eval.sh" --ckpt released --custom "$CUSTOM" --out "$OUT/zeroshot" "${eval_opt[@]}"
fi
for r in $RATIOS; do
    d="$OUT/ratio$r"; runs+=("$d")
    [ -f "$d/results/summary.json" ] && { echo "skip $d (done)"; continue; }
    bash "$HERE/train.sh" --custom "$CUSTOM" --ratio "$r" --out "$d" "${pass[@]}"
done
source "$HERE/env.sh"
python "$VDV/tools/export_metrics.py" compare "${runs[@]}" --to "$OUT/compare.md"
