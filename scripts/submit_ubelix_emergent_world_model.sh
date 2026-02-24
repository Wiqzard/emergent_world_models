#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/outputs/ubelix_world_model}"
PARTITION="${PARTITION:-gpu}"
CONDA_ENV="${CONDA_ENV:-emergent-multiagent}"
JOB_TIME="${JOB_TIME:-08:00:00}"
GPU_PREFERENCE="${GPU_PREFERENCE:-rtx4090,l40s,a100,rtx3090}"

EPOCHS="${EPOCHS:-50}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-40}"
EVAL_BATCHES="${EVAL_BATCHES:-8}"
SEQ_LEN="${SEQ_LEN:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PIXEL_HORIZON="${PIXEL_HORIZON:-8}"
PIXEL_EVAL_EVERY="${PIXEL_EVAL_EVERY:-5}"

WANDB_FLAG=""
if [[ "${WANDB:-0}" == "1" ]]; then
  WANDB_FLAG="--wandb --wandb-project ${WANDB_PROJECT:-emergent-world-models}"
  if [[ -n "${WANDB_RUN_NAME:-}" ]]; then
    WANDB_FLAG+=" --wandb-run-name ${WANDB_RUN_NAME}"
  fi
fi

mkdir -p "$OUT_DIR"

echo "== sinfo gpu summary =="
sinfo -N -h -o "%N %P %G" | awk '$3 != "(null)"'

selected_type=""
selected_count=0
IFS=',' read -r -a gpu_types <<< "$GPU_PREFERENCE"
for gpu_type in "${gpu_types[@]}"; do
  for n in 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1; do
    if sbatch --test-only \
      --partition="$PARTITION" \
      --nodes=1 \
      --gpus-per-node="${gpu_type}:${n}" \
      --ntasks=1 \
      --cpus-per-task=4 \
      --mem=8G \
      --time=00:10:00 \
      --wrap hostname >/tmp/emergent_sbatch_test.out 2>/tmp/emergent_sbatch_test.err; then
      selected_type="$gpu_type"
      selected_count="$n"
      break 2
    fi
  done
done

if [[ "$selected_count" -eq 0 ]]; then
  echo "No allocatable GPU type/count found in partition ${PARTITION}."
  cat /tmp/emergent_sbatch_test.err || true
  exit 1
fi

echo "Using ${selected_type}:${selected_count}"

module load Anaconda3
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  conda env create -n "$CONDA_ENV" -f "$ROOT_DIR/environment.yml"
fi
if ! conda run -n "$CONDA_ENV" python -c "import minigrid" >/dev/null 2>&1; then
  conda run -n "$CONDA_ENV" pip install minigrid
fi

SBATCH_FILE="$OUT_DIR/run_world_model_${selected_type}_${selected_count}.sbatch"
cat > "$SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=emergent_wm
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=${selected_type}:${selected_count}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=${JOB_TIME}
#SBATCH --output=${OUT_DIR}/slurm-%j.out

set -euo pipefail
module load Anaconda3
eval "\$(conda shell.bash hook)"
cd ${ROOT_DIR}
conda run -n ${CONDA_ENV} python gym_distributed_observer_direct_pixel_eval.py \
  --env MiniGrid-Dynamic-Obstacles-16x16-v0 \
  --graph sphere \
  --graph-rows 8 \
  --graph-cols 4 \
  --observer-placement cluster2d \
  --epochs ${EPOCHS} \
  --steps-per-epoch ${STEPS_PER_EPOCH} \
  --eval-batches ${EVAL_BATCHES} \
  --seq-len ${SEQ_LEN} \
  --batch-size ${BATCH_SIZE} \
  --pixel-horizon ${PIXEL_HORIZON} \
  --pixel-eval-every ${PIXEL_EVAL_EVERY} \
  --save-pixel-mp4 \
  --pixel-mp4-prefix ${OUT_DIR}/pixel_pred \
  --pixel-plot-file ${OUT_DIR}/pixel_pred.png \
  ${WANDB_FLAG}
EOF

sbatch "$SBATCH_FILE"
