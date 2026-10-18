# Source me: shared environment for all video_data_verify scripts.
VDV="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]:-$0}")")")"  # this codebase
WS="$(dirname "$VDV")"                    # workspace: lite_model, envs, ckpts, data, outputs
LITE="$WS/lite_model"
export WS VDV LITE
export PATH="$WS/envs/lite/bin:$PATH"
export PYTHONPATH="$VDV:$LITE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-$WS/.cache/huggingface}   # keep if already set
export TORCH_HOME=${TORCH_HOME:-$WS/.cache/torch}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export WANDB_MODE=${WANDB_MODE:-disabled}
export PYTHONWARNINGS=${PYTHONWARNINGS:-ignore}
# Data, released checkpoint and the default parent of run directories.
export VDV_DATA=${VDV_DATA:-$WS/data}
# Standard set: a manifest (10k frames sampled uniformly from the SpatialEncoder training
# data + the CA-1M test set), or any single dataset root in the vdv format.
export VDV_STD=${VDV_STD:-$VDV/data/standard_10k.json}
export VDV_OUT=${VDV_OUT:-$WS/outputs/video_data_verify}
export CKPT_RELEASED=${CKPT_RELEASED:-$WS/ckpts/se_small_lite_wds518_e100/checkpoint.pt}

# vdv_launch NGPUS ARGS...: one vdv.run process per GPU (CUDA_VISIBLE_DEVICES picks them).
vdv_launch() {
    local n=$1; shift
    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$n" \
        -m vdv.run "$@"
}
