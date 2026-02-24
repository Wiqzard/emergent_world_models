# AGENTS.md

## Project context
This repo contains lightweight, reproducible experiments for emergent world models in multi‑agent settings.

## Conventions
- Keep scripts runnable from the repo root with relative paths only. Do not hardcode absolute paths.
- If you add or change dependencies, update `environment.yml` and mention it in `README.md`.
- Prefer simple, readable PyTorch code over heavy frameworks.
- Default outputs should be printed to stdout; avoid writing large artifacts unless requested.

## Environment
- Primary environment: conda env `emergent-multiagent`.
- Default Python: 3.10.

## Reproducibility
- Always set random seeds for NumPy, Python, and PyTorch.
- Keep default hyperparameters small enough to run on CPU.

## When editing
- Preserve existing CLI flags unless a change is explicitly requested.
- Document new flags in `README.md`.
- After completing requested code/doc changes, commit and push to GitHub automatically unless explicitly told not to.

## UBELIX (SLURM)
- Use UBELIX as the default compute system for this project.
- Connect using `ssh ss24i671@submit03.unibe.ch`.
- Work through SLURM only (`srun` interactive or `sbatch` batch).
- Always request the maximum number of GPUs available in one allocation for the chosen GPU type.
  - Prefer `rtx4090` if available.
  - Otherwise use the top available GPU type.
- Before submitting jobs, inspect resources and limits, then set `--gpus-per-node` to the max allowed for that type.
- Run experiments from `~/Documents/emergent_world_models` on UBELIX.
- Standard UBELIX entrypoint:
  - `bash scripts/submit_ubelix_emergent_world_model.sh`
  - This script auto-detects max allocatable GPU count (prefers `rtx4090`), creates/uses `emergent-multiagent`, and submits the world-model experiment job.

Reference flow:

```bash
ssh ss24i671@submit03.unibe.ch
cd ~/Documents
git clone https://github.com/Wiqzard/emergent_world_models.git
cd emergent_world_models

# Check available GPU resources (pick maximum allocatable count)
sinfo -N -h -o "%N %P %G"

# Example interactive allocation
srun --partition=gpu --gpus-per-node=rtx4090:<MAX_COUNT> --time=08:00:00 --pty bash

module load Anaconda3
eval "$(conda shell.bash hook)"
conda env create -f environment.yml
conda run -n emergent-multiagent python gym_distributed_observer_direct_pixel_eval.py --env MiniGrid-Dynamic-Obstacles-16x16-v0 --epochs 50
```
