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

Shared-weights efficient single-observer variant (one shared agent over flattened agent batch, with per-agent identity embeddings; observer predicts token, separate predictor agent predicts neighbor next states):

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
--predictor-agent 1
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
