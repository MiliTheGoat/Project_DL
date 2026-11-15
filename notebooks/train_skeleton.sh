#!/bin/bash
# =========================================================================================================
# Train and test CTR-GCN (2D skeletons) on the DTU HPC cluster.
#
# Always start it from the project root (the folder that holds configs/, src/ and notebooks/):
#     bash notebooks/train_skeleton.sh setup          FIRST TIME, on a login node: builds the Python environment,
#                                                     unpacks HMDB, downloads the pose model. No training.
#     bsub < notebooks/train_skeleton.sh              submit the job for experiment exp1
#     EXP=exp2 bsub < notebooks/train_skeleton.sh     submit the job for another experiment yaml
#
# While it runs:   bstat                   is the job waiting or running?
#                  tail -f hpc_skeleton_<JOBID>.out      what the script prints
#                  bnvtop <JOBID>          is the GPU really used?        bkill <JOBID>   stop the job
# Results:         models/hpc_runs/ctr_gcn_2d_<EXP>/   (best.pt, history.csv, summary.json, test_results*.json)
# If the job is stopped (time limit, crash), submit it again: training continues from the last finished epoch.
# =========================================================================================================
#BSUB -q gpuv100                               # Tesla V100 queue.  A100 instead: gpua100
#BSUB -J ctr_gcn_skeleton                      # job name
#BSUB -gpu "num=1:mode=exclusive_process"      # one GPU, for this job alone
#BSUB -n 4                                     # 4 CPU cores (the minimum the GPU queues accept)
#BSUB -R "span[hosts=1]"                       # all cores on the same machine
#BSUB -R "rusage[mem=8GB]"                     # memory PER CORE -> 4 x 8 GB = 32 GB
#BSUB -W 6:00                                  # time limit hh:mm (the queue allows at most 24:00)
#BSUB -o hpc_skeleton_%J.out                   # what the script prints   (%J = job number)
#BSUB -e hpc_skeleton_%J.err                   # error messages

set -e                                         # stop at the first step that fails

EXP="${EXP:-exp1}"                             # which yaml in configs/experiments/skeleton_experiments/
RUN_NAME="ctr_gcn_2d_${EXP}"                   # results go to models/hpc_runs/<RUN_NAME>/
EXP_DIR="configs/experiments/skeleton_experiments"
HPC_EXP="hpc_${EXP}"                           # copy of the experiment yaml with the cluster settings
SRC="src/rgb_vs_skeleton"
PYTHON=".venv/bin/python"                      # on the cluster ALWAYS use this python, never "uv run" (see step 1)
NTU_SKELETONS="data/processed/skeleton_cache_2d13.npz"                  # made on your PC, upload it
HMDB_SKELETONS="data/external/skeleton_estm/hmdb_skeleton_2d13.npz"     # made in step 5 if it is not there
POSE_MODEL="yolo11s-pose.pt"                   # same as pose_model in configs/main_config.yaml
export PATH="$HOME/.local/bin:$PATH"           # where uv is installed

if [ ! -f configs/main_config.yaml ]; then echo "Start this script from the project root folder."; exit 1; fi
if [ ! -f "$EXP_DIR/$EXP.yaml" ]; then echo "Experiment file $EXP_DIR/$EXP.yaml does not exist."; exit 1; fi
echo "===== CTR-GCN (2D skeletons), experiment $EXP -> models/hpc_runs/$RUN_NAME   $(date)"


echo; echo "===== 1. Python environment"
# The project's uv.lock installs the CPU-only PyTorch (right for a laptop, useless on a GPU node). So: install
# everything from uv.lock EXCEPT torch, then add the same torch version built for the GPU (CUDA 12.4).
# "uv run" would put the CPU version back, which is why this script calls .venv/bin/python directly.
if ! command -v uv > /dev/null; then
    echo "installing uv into ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
if [ ! -f .venv/hpc_ready ]; then
    echo "building the environment in .venv (only the first time, a few minutes)"
    uv sync --no-dev --no-cache --no-install-package torch --no-install-package torchvision
    uv pip install --no-cache --python "$PYTHON" torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124
    touch .venv/hpc_ready
else
    echo "environment found: .venv"
fi
if ! $PYTHON -c "import cv2" 2> /dev/null; then     # some cluster nodes lack a graphics library OpenCV wants
    echo "OpenCV does not start here: switching to the version without graphics"
    uv pip uninstall --python "$PYTHON" opencv-python
    uv pip install --no-cache --python "$PYTHON" opencv-python-headless
fi


echo; echo "===== 2. Data"
# NTU skeletons: one file with all clips already in the 2D 13-joint format. The original .skeleton files are not needed.
if [ ! -f "$NTU_SKELETONS" ]; then echo "MISSING: upload $NTU_SKELETONS first."; exit 1; fi
if [ ! -f data/processed/splits.csv ]; then echo "MISSING: data/processed/splits.csv (it is in git: git pull)."; exit 1; fi
echo "NTU skeletons:  $NTU_SKELETONS ($(du -h "$NTU_SKELETONS" | cut -f1))"

# HMDB skeletons: either they are already there, or step 5 makes them from the HMDB videos.
if [ -f "$HMDB_SKELETONS" ]; then
    HMDB_READY=yes
    echo "HMDB skeletons: $HMDB_SKELETONS found, pose estimation will be skipped"
else
    if [ ! -d data/external/processed/drink ] && [ -f data/external/processed.zip ]; then
        echo "unpacking data/external/processed.zip ..."
        unzip -q -n data/external/processed.zip -d data/external/
    fi
    if [ -d data/external/processed/drink ]; then
        HMDB_READY=estimate
        echo "HMDB skeletons: not there yet, step 5 makes them from the $(find data/external/processed -name '*.avi' | wc -l) HMDB videos"
        # the pose model is downloaded once into the project folder (*.pt files are ignored by git)
        if ! $PYTHON -c "from ultralytics import YOLO; YOLO('$POSE_MODEL')" > /dev/null 2>&1; then
            echo "Could not load or download $POSE_MODEL. Run 'bash notebooks/train_skeleton.sh setup' on a login node."
            exit 1
        fi
        echo "pose model:     $POSE_MODEL is available"
    else
        HMDB_READY=no
        echo "HMDB skeletons: NOT FOUND, and no HMDB videos to make them from -> the HMDB test will be skipped"
        echo "                (upload $HMDB_SKELETONS or data/external/processed.zip)"
    fi
fi

if [ "$1" = "setup" ]; then
    echo; echo "===== Setup finished. Submit the job with:  bsub < notebooks/train_skeleton.sh"
    exit 0
fi


echo; echo "===== 3. GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PYTHON -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see a GPU'; print('PyTorch', torch.__version__, 'uses', torch.cuda.get_device_name(0))"


echo; echo "===== 4. Experiment settings for the cluster"
# wandb: live charts need internet on the compute node and a login ('wandb login' once on the login node).
# Without them the run is stored in ./wandb and you upload it later with:  .venv/bin/wandb sync wandb/offline-run-*
WANDB=offline
if curl -s --max-time 10 https://api.wandb.ai > /dev/null 2>&1; then
    if [ -n "$WANDB_API_KEY" ] || grep -q "api.wandb.ai" "$HOME/.netrc" 2> /dev/null; then WANDB=online; fi
fi
# A copy of the experiment yaml: same hyper-parameters, but the cluster's name, output folder and worker count.
# restart is set to false so that a job that is submitted again continues instead of starting over.
sed -e "s|^name:.*|name: ${RUN_NAME}|" \
    -e "s|^output_dir:.*|output_dir: models/hpc_runs|" \
    -e "s|^num_workers:.*|num_workers: 4|" \
    -e "s|^restart:.*|restart: false|" \
    -e "s|^wandb_mode:.*|wandb_mode: ${WANDB}|" \
    "$EXP_DIR/$EXP.yaml" > "$EXP_DIR/$HPC_EXP.yaml"
echo "wrote $EXP_DIR/$HPC_EXP.yaml (wandb: $WANDB)"


echo; echo "===== 5. HMDB skeletons (pose estimation)   $(date)"
if [ "$HMDB_READY" = "estimate" ]; then
    $PYTHON $SRC/pose_estimation.py
    HMDB_READY=yes
else
    echo "nothing to do"
fi

echo; echo "===== 6. TRAIN   $(date)"
$PYTHON $SRC/train_skeleton.py +exp=$HPC_EXP

echo; echo "===== 7. TEST on the NTU test set   $(date)"
$PYTHON $SRC/evaluate_skeleton.py +exp=$HPC_EXP

echo; echo "===== 8. TEST on the HMDB-51 skeletons (cross-dataset)   $(date)"
if [ "$HMDB_READY" = "yes" ]; then
    $PYTHON $SRC/evaluate_skeleton.py +exp=$HPC_EXP +test_on=hmdb
else
    echo "skipped: no HMDB skeletons"
fi

echo; echo "===== DONE in $((SECONDS / 60)) minutes. Results: models/hpc_runs/$RUN_NAME   $(date)"
ls -lh "models/hpc_runs/$RUN_NAME"