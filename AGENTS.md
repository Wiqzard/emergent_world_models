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
