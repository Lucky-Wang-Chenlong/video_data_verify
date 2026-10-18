#!/usr/bin/env bash
# Finetune the released SpatialEncoder lite checkpoint on the standard data
# (CA-1M 10k) mixed with a custom video dataset, validating on both test sets.
#
#   scripts/train.sh --custom DIR [--ratio R] [--out DIR] [options] [key=value ...]
#
#   --custom DIR        custom dataset directory (see vdv/README.md)           [required]
#   --ratio R           fraction of training clips drawn from the custom train split [0.2]
#                       0 = standard data only (control, A1); 1 = custom only
#   --out DIR           output directory                [$VDV_OUT/<dataset>_ratio<R>]
#   --gpus N            number of GPUs (CUDA_VISIBLE_DEVICES selects them)         [8]
#   --epochs N          epochs (cosine LR over all of them)                         [5]
#   --samples N         global training clips per epoch, same for every ratio    [1600]
#   --lr X              head learning rate (backbone / spatial encoder: 1e-7)    [3e-5]
#   --train-split NAME  custom train split tag in index.json                    [train]
#   --test-split NAME   custom test split tag in index.json                      [test]
#   --depth-scale S     custom PNG depth units per meter (1000 = millimeters)    [1000]
#   --val-every N       validate every N epochs; always after the last            [1]
#   --workers N         dataloader workers per GPU                                [4]
#   --init-ckpt CKPT    start from this checkpoint              [released lite checkpoint]
#   --overfit           sanity check: custom data only, 10 epochs x 800 clips,
#                       validated on the custom *train* split
#   key=value           extra lite_model config overrides, e.g. scratch.frame_num_range=[2,4]
#
# The validation after the last epoch uses the final weights on both full test
# sets; it is the result of the run (results/summary.md). Re-running with the
# same --out resumes an interrupted run.
#
# Output (--out):
#   checkpoints/checkpoint.pt   final weights (also the resume point)
#   metrics/train.csv           loss, loss terms and LR every 10 steps
#   metrics/val.csv             per validation and test set: miou, acc10/25/50, errors
#   metrics/val_per_category.csv
#   results/summary.md|json     final metrics, overall and per category
#   logs/                       full logs (log.txt, train/val_stats.json); tensorboard/
#   config.yaml                 complete resolved lite_model config of the run
set -euo pipefail
HERE="$(dirname "$(realpath "$0")")"
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

CUSTOM=""; RATIO=0.2; OUT=""; GPUS=8; EPOCHS=""; SAMPLES=""; LR=""; TRAIN=train; TEST=test
DEPTH_SCALE=1000; VAL_EVERY=1; WORKERS=""; INIT=""; OVERFIT=0; extra=()
while [ $# -gt 0 ]; do
    case $1 in
        --custom) CUSTOM=$2; shift 2 ;;
        --ratio) RATIO=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --gpus) GPUS=$2; shift 2 ;;
        --epochs) EPOCHS=$2; shift 2 ;;
        --samples) SAMPLES=$2; shift 2 ;;
        --lr) LR=$2; shift 2 ;;
        --train-split) TRAIN=$2; shift 2 ;;
        --test-split) TEST=$2; shift 2 ;;
        --depth-scale) DEPTH_SCALE=$2; shift 2 ;;
        --val-every) VAL_EVERY=$2; shift 2 ;;
        --workers) WORKERS=$2; shift 2 ;;
        --init-ckpt) INIT=$2; shift 2 ;;
        --overfit) OVERFIT=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *=*) extra+=("$1"); shift ;;
        *) echo "unknown option: $1" >&2; usage; exit 1 ;;
    esac
done
[ -n "$CUSTOM" ] || { echo "--custom DIR is required" >&2; usage; exit 1; }
CUSTOM="$(realpath "$CUSTOM")"
[ -z "$OUT" ] || OUT="$(realpath -m "$OUT")"
[ -z "$INIT" ] || INIT="$(realpath "$INIT")"
source "$HERE/env.sh"
INIT=${INIT:-$CKPT_RELEASED}
python - "$VDV_STD" <<'EOF' || { echo "standard data missing ($VDV_STD): run scripts/download_data.sh" >&2; exit 1; }
import os, sys
from vdv.config import load_standard
std = load_standard(sys.argv[1])
missing = [e["root"] for e in std["train"] + std["test"] if not os.path.exists(os.path.join(e["root"], "index.json"))]
sys.exit(f"missing: {missing}" if missing else 0)
EOF

VAL_SETS=all
if [ "$OVERFIT" = 1 ]; then
    RATIO=1; EPOCHS=${EPOCHS:-10}; SAMPLES=${SAMPLES:-800}; TEST=$TRAIN; VAL_SETS=custom
fi
NAME="$(basename "$CUSTOM")_$([ "$OVERFIT" = 1 ] && echo overfit || echo "ratio$RATIO")"
OUT=${OUT:-$VDV_OUT/$NAME}
if [ -f "$OUT/results/summary.json" ]; then
    echo "$OUT already holds a finished run; pick another --out" >&2; exit 1
fi
mkdir -p "$OUT/logs"
opt=()
[ -z "$EPOCHS" ] || opt+=(--epochs "$EPOCHS")
[ -z "$SAMPLES" ] || opt+=(--samples "$SAMPLES")
[ -z "$LR" ] || opt+=(--lr "$LR")
[ -z "$WORKERS" ] || opt+=(--workers "$WORKERS")

echo "$(date +%T) train $NAME: custom=$CUSTOM ($TRAIN / $TEST) ratio=$RATIO gpus=$GPUS -> $OUT"
set +e
vdv_launch "$GPUS" --mode train --out "$OUT" --std "$VDV_STD" --custom "$CUSTOM" \
    --ratio "$RATIO" --init-ckpt "$INIT" --train-split "$TRAIN" --test-split "$TEST" \
    --depth-scale "$DEPTH_SCALE" --val-sets "$VAL_SETS" --val-every "$VAL_EVERY" \
    "${opt[@]}" "${extra[@]}" >> "$OUT/logs/train.log" 2>&1
rc=$?
set -e
python "$VDV/tools/export_metrics.py" run "$OUT" || true
[ "$rc" -eq 0 ] || { echo "training failed ($rc), see $OUT/logs/train.log" >&2; exit "$rc"; }
[ -f "$OUT/results/summary.md" ] && cat "$OUT/results/summary.md" | sed -n '1,12p'
echo "$(date +%T) done: $OUT"
