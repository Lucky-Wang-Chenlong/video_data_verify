#!/usr/bin/env bash
# One-time setup of the workspace (the parent directory of this repo):
#   1. lite_model (SpatialEncoder v2 Small) next to this codebase
#   2. Python 3.10 venv with the lite_model training requirements
#   3. released lite checkpoint from HuggingFace (gated: `hf auth login` first)
# Data is downloaded separately: scripts/download_data.sh.
set -euo pipefail
VDV="$(dirname "$(realpath "$0")")"
WS="$(dirname "$VDV")"
LITE="$WS/lite_model"
ENV="$WS/envs/lite"
mkdir -p "$WS/envs"
exec > >(tee -a "$WS/envs/setup_env.log") 2>&1

export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$WS/.cache/pip}
export HF_HOME=${HF_HOME:-$WS/.cache/huggingface}
PYTHON=${PYTHON:-python3.10}

# ---- 1. repo -------------------------------------------------------------
[ -d "$LITE" ] || git clone https://github.com/qic999/spatial_encoder_v2_small.git "$LITE"

# ---- 2. env (lite_model README recipe) -----------------------------------
[ -x "$ENV/bin/python" ] || "$PYTHON" -m venv "$ENV"
PY="$ENV/bin/python"
$PY -m pip install -U pip wheel
$PY -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install -r "$LITE/requirements/lite-migration.txt"
$PY -m pip install -e "$LITE[train]"
$PY -m pip install --no-deps 'git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'
$PY -m pip install -r "$LITE/requirements/perception-encoder.txt"
$PY -m pip install psutil   # imported by sam3, missing from the requirement files
$PY -c "import torch, sam3, depth_anything_3; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ---- 3. weights ----------------------------------------------------------
mkdir -p "$WS/ckpts"
$PY - "$WS/ckpts" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download("SpatialCoding/Data_verification", "se_small_lite_wds518_e100/checkpoint.pt",
                      local_dir=sys.argv[1]))
EOF
echo "setup done: env $ENV, checkpoint $WS/ckpts/se_small_lite_wds518_e100/checkpoint.pt"
