# Emergent Multi-Agent Neighbor Prediction

This repo contains a simple multi-agent experiment where agents receive partial observations and predict the next internal state of their neighbors.

## Environment setup

Create the conda environment:

```bash
conda env create -f environment.yml
```

Activate it (if your conda registry isn't writable, activate by path):

```bash
conda activate emergent-multiagent
```

If you already created the env before newer experiments were added, update dependencies:

```bash
conda env update -f environment.yml --prune
```

## Run

```bash
python experiment.py
```

## Notes

- Default env is `CartPole-v1`.
- 16 agents, ring graph.
- Logs MSE for observers, non-observers, and overall.
- A random action is sampled each step and provided only to observer agents for next-state prediction.

## Useful flags

```bash
--env CartPole-v1
--agents 16
--graph ring|ring-k|line|full|random|dense|grid|star
--neighbors 4
--degree 4
--observer-frac 0.5
--min-obs-dims 1
--state-dim 16
--hidden-dim 64
--hidden-layers 2
--episodes 200
--steps-per-episode 200
--lr 3e-4
--seed 7
--log-interval 10
--device auto|cpu|cuda|mps
```

## Graph notes

- `ring`: each agent has 2 neighbors (left/right).
- `ring-k`: each agent has `--neighbors` neighbors split evenly left/right (must be even).
- `line`: chain; endpoints have 1 neighbor.
- `full`: each agent connects to all others.
- `random`: each agent connects to `--neighbors` random neighbors.
- `dense`: each agent connects to the next `--neighbors` agents in cyclic order.
- `grid`: 2D grid with 4-neighborhood (requires perfect square number of agents).
- `star`: agent 0 is hub; all others connect to it.

## Language experiment

A small language-modeling variant with a causal rollout over agent index. Within each sampled window of length `agents+1`, step `k` reveals tokens only to observer agents `<= k` (later observers still receive zero input). At each step, every agent forms its new latent state from its current input plus all neighbor latent states from the previous rollout step. Agent `k` then predicts token `t_{k+1}`. Non-observers are trained with neighbor-state prediction. Sentences must be long enough for the chosen number of agents (or use `--sequence-mode stream` to concatenate).

```bash
python language_experiment.py
```

Efficient causal implementation (same objective, less repeated encoding work per step):

```bash
python language_experiment_efficient.py
```

Single-observer recurrent variant (observer gets token; all agents predict neighbor next states; observer also predicts next token):

```bash
python language_experiment_single_observer.py
```

Shared-weights efficient single-observer variant (one shared agent over flattened agent batch, with per-agent identity embeddings):

```bash
python language_experiment_single_observer_shared.py
```

Useful flags:

```bash
--corpus-file /path/to/text.txt
--embed-dim 32
--state-dim 32
--hidden-dim 64
--hidden-layers 2
--batch-size 32
--epochs 100
--steps-per-epoch 200
--sequence-mode stream|sentence
```

Additional flags for the single-observer variant:

```bash
--observer-agent 0
--unroll-steps 16
```

Additional performance flags for the shared variant:

```bash
--compile auto|on|off
--amp auto|on|off
--eval-batches 8
```

Additional dataset options (requires torchtext for real datasets):

```bash
--dataset toy|wikitext2|ptb
--dataset-root /path/to/cache
--dataset-split train|valid|test
```

Torchtext install:

```bash
conda install -c conda-forge torchtext
```

## Distributed Local World Model experiment

New experiment for your hypothesis: agents only optimize local predictive objectives (own patch + neighbors), then we test whether all agent latents jointly encode the full environment.

```bash
python distributed_local_world_model_experiment.py
```

What it does:

- Environment: synthetic diffusion field on a 2D grid (local coupling only).
- Agents: one per grid cell, 4-neighbor communication graph.
- Blindness is first-class: each identity has an observation mask that is fed into the recurrent update.
- Learnable identity embeddings are used in latent initialization, updates, and message functions.
- Messages are identity-conditioned and include relative edge features (`dx`, `dy`).
- Training losses:
  - sensing agents predict their own next local patch (`--lambda-self`).
  - all agents predict neighbors' next latent states (`--lambda-neighbor`).
- Optional blind-latent stability regularizer (`--blind-reg-weight`).
- Optional anti-cheat setup: permute identity-to-position assignment each rollout (`--permute-agent-positions`).
- Evaluation: freeze the model, train a linear probe from all agent latents to the full global grid state.
- Reports global probe MSE/R2 plus reconstruction split on observed vs blind cells.
- Optional per-agent probe quality is available when positions are not permuted (`--skip-agent-probes` to disable).

Useful flags:

```bash
--grid-size 8
--patch-radius 1
--observer-frac 0.5
--latent-dim 16
--id-dim 8
--epochs 40
--steps-per-epoch 40
--lambda-self 1.0
--lambda-neighbor 1.0
--blind-reg-weight 1e-3
--disable-messages          # ablation: no communication
--permute-agent-positions   # anti-cheat: random identity-to-position assignment
--probe-train-batches 24
--probe-test-batches 12
--agent-probe-max-agents 0
```

## Gym Distributed World Model experiment (with random actions, visualization, and W&B)

This variant uses a Gym environment (default: `Acrobot-v1`) and makes the signal flow explicit:

- At each macro-step, random actions are sampled from the environment action space.
- `--frame-skip` controls how many env steps are executed per macro-step.
- Only observer agents receive:
  - their assigned part of the current state (`--min-obs-dims` controls minimum visible dimensions),
  - an action vector aggregated over the executed actions in that macro-step (same action dimensionality).
- Blind agents receive no direct state/action input and must rely on neighbor messages.
- The model is still trained only with local objectives (self local next-state prediction + neighbor latent prediction).
- After training, a frozen global probe predicts full next state from all agents' latents.

Run:

```bash
python gym_distributed_local_world_model_experiment.py
```

W&B logging:

```bash
python gym_distributed_local_world_model_experiment.py --wandb --wandb-project emergent-world-models
```

Visualization:

- Saves a metrics figure to `outputs/gym_world_model_metrics.png` by default (`--plot-file` to override).
- The plot includes:
  - train/eval loss curves,
  - global probe test MSE vs baseline,
  - observer-vs-blind single-agent probe MSE (if enabled).
- Optional pixel-level visualization (`--pixel-probe`) fits an additional latent->RGB-frame probe and saves:
  - `outputs/gym_pixel_prediction_comparison.png` (or `--pixel-plot-file`),
  - row 1 = true next frames, row 2 = predicted next frames.
- `--pixel-horizon` controls how many macro-steps ahead the pixel target is (`t+K`).
- For classic-control pixel rendering (`CartPole-v1`, `Acrobot-v1`), `pygame` is required (now included in `environment.yml`).

Pixel comparison run example:

```bash
python gym_distributed_local_world_model_experiment.py --env CartPole-v1 --pixel-probe
```

Useful flags:

```bash
--env Acrobot-v1
--agents 32
--graph ring|line
--observer-frac 0.5
--min-obs-dims 2
--latent-dim 32
--id-dim 8
--batch-size 16
--seq-len 10
--frame-skip 4
--epochs 50
--steps-per-epoch 40
--lambda-self 1.0
--lambda-neighbor 1.0
--blind-reg-weight 1e-3
--disable-messages
--permute-agent-positions
--probe-train-batches 24
--probe-test-batches 12
--plot-file outputs/gym_world_model_metrics.png
--pixel-probe
--pixel-probe-train-batches 12
--pixel-probe-test-batches 6
--pixel-probe-batch-size 4
--pixel-height 84
--pixel-width 84
--pixel-horizon 4
--pixel-plot-file outputs/gym_pixel_prediction_comparison.png
--wandb
```
