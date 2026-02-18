# Emergent Multi-Agent Neighbor Prediction

This repo contains a simple multi-agent experiment where agents receive partial observations and predict the next internal state of their neighbors.

## Environment setup

Create the conda environment:

```bash
conda env create -f "/Users/sebastianstapf/Documents/emergent_world_models/environment.yml"
```

Activate it (if your conda registry isn't writable, activate by path):

```bash
conda activate /opt/homebrew/anaconda3/envs/emergent-multiagent
```

## Run

```bash
conda run -n emergent-multiagent python "/Users/sebastianstapf/Documents/emergent_world_models/experiment.py"
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

A small language-modeling variant with an RNN-style alignment: agent i sees token t_i and predicts token t_{i+1} within a sampled window of length agents+1. Observer agents receive their token; non-observers get zero input but still predict for evaluation. Sentences must be long enough for the chosen number of agents.

```bash
conda run -n emergent-multiagent python "/Users/sebastianstapf/Documents/emergent_world_models/language_experiment.py"
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
