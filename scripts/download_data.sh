#!/usr/bin/env bash
# Download the standard data and build the toy custom set.
#
#   scripts/download_data.sh [DATA_DIR] --accept-license        [DATA_DIR = $WS/data]
#
# Standard data (data/standard_10k.json): 10,000 training frames sampled uniformly
# from the SpatialEncoder training set (HF qicq1c/spatialencoder_full; fixed plan in
# data/standard_10k_plan.json) + the CA-1M test set. CA-1M is released under
# CC-BY-NC-ND 4.0 (non-commercial); --accept-license confirms you accept it.
# Finished steps are skipped, so re-running resumes. Downloads ~90 GB of archives
# (only the sampled frames are kept, archives are deleted); ~12 GB on disk afterwards:
#   DATA_DIR/standard_10k/{ca1m,adt,hypersim}/  sampled training windows, one vdv root per source
#                             (CA-1M 9,216 / ADT 500 / Hypersim 200 = 9,916 frames)
#   DATA_DIR/ca1m/raw/        the 2 CA-1M val captures of data/ca1m_test.txt
#   DATA_DIR/ca1m/vdv_test/   their vdv version (split test: 2,131 frames, 270 test clips)
#   DATA_DIR/ca1m/raw_toy/    3 CA-1M train captures (excluded from the standard sample)
#   DATA_DIR/custom/toy/      toy custom set: their "cushion" boxes (format example, smoke tests)
set -euo pipefail
HERE="$(dirname "$(realpath "$0")")"
D=""; ACCEPT=""
for a in "$@"; do
    case $a in
        --accept-license) ACCEPT=--accept-license ;;
        -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        *) D=$a ;;
    esac
done
source "$HERE/env.sh"
D="$(realpath -m "${D:-$VDV_DATA}")"
[ -n "$ACCEPT" ] || { echo "CA-1M is CC-BY-NC-ND 4.0; re-run with --accept-license to accept it" >&2; exit 1; }
echo "data dir: $D"
mkdir -p "$D/ca1m"

# Standard training set: uniform 10k-frame sample of the SpatialEncoder training data
python "$VDV/tools/sample_standard.py" build --plan "$VDV/data/standard_10k_plan.json" \
    --out "$D/standard_10k"

# CA-1M test set
python "$VDV/tools/download_ca1m.py" --list "$VDV/data/ca1m_test.txt" --raw "$D/ca1m/raw" $ACCEPT
[ -f "$D/ca1m/vdv_test/index.json" ] || python "$VDV/tools/prepare_ca1m.py" \
    --raw "$D/ca1m/raw" --out "$D/ca1m/vdv_test" --list "$VDV/data/ca1m_test.txt"

# Toy custom set
python "$VDV/tools/make_toy_custom.py" --raw - --out - --print-list > "$D/ca1m/toy_list.txt"
python "$VDV/tools/download_ca1m.py" --list "$D/ca1m/toy_list.txt" --raw "$D/ca1m/raw_toy" $ACCEPT
[ -f "$D/custom/toy/index.json" ] || python "$VDV/tools/make_toy_custom.py" \
    --raw "$D/ca1m/raw_toy" --out "$D/custom/toy"
[ "$D" = "$VDV_DATA" ] || echo "note: export VDV_DATA=$D for the other scripts"
