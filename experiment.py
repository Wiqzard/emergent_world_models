import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    import gymnasium as gym
except Exception:  # gymnasium not installed
    import gym


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Graph:
    neighbors: List[List[int]]


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, hidden_layers: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Agent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        num_neighbors: int,
        action_dim: int,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
    ):
        super().__init__()
        self.encoder = MLP(obs_dim, state_dim, hidden_dim, hidden_layers)
        pred_in = state_dim * (1 + num_neighbors) + action_dim
        pred_out = state_dim * num_neighbors
        self.predictor = MLP(pred_in, pred_out, hidden_dim, hidden_layers)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def predict_neighbors(
        self,
        self_state: torch.Tensor,
        neighbor_states: torch.Tensor,
        action_vec: torch.Tensor,
    ) -> torch.Tensor:
        # neighbor_states: [B, num_neighbors, state_dim]
        bsz = self_state.shape[0]
        flat_neighbors = neighbor_states.reshape(bsz, -1)
        x = torch.cat([self_state, flat_neighbors, action_vec], dim=-1)
        pred = self.predictor(x)
        return pred.reshape(bsz, -1, self_state.shape[-1])



def build_graph(num_agents: int, graph_type: str, neighbors: int) -> Graph:
    if neighbors < 1 and graph_type not in {"full"}:
        raise ValueError("neighbors must be >= 1 for non-full graphs")
    if neighbors > num_agents - 1 and graph_type not in {"full"}:
        raise ValueError("neighbors must be <= num_agents - 1")

    if graph_type == "ring":
        neighbors = []
        for i in range(num_agents):
            neighbors.append([ (i - 1) % num_agents, (i + 1) % num_agents ])
        return Graph(neighbors)
    if graph_type == "ring-k":
        if neighbors % 2 != 0:
            raise ValueError("ring-k requires an even --neighbors value")
        half = neighbors // 2
        out = []
        for i in range(num_agents):
            neigh = []
            for d in range(1, half + 1):
                neigh.append((i - d) % num_agents)
                neigh.append((i + d) % num_agents)
            out.append(neigh)
        return Graph(out)
    if graph_type == "line":
        out = []
        for i in range(num_agents):
            neigh = []
            if i - 1 >= 0:
                neigh.append(i - 1)
            if i + 1 < num_agents:
                neigh.append(i + 1)
            out.append(neigh)
        return Graph(out)
    if graph_type == "full":
        neighbors = []
        for i in range(num_agents):
            neighbors.append([j for j in range(num_agents) if j != i])
        return Graph(neighbors)
    if graph_type == "random":
        out = []
        for i in range(num_agents):
            choices = [j for j in range(num_agents) if j != i]
            out.append(random.sample(choices, k=min(neighbors, len(choices))))
        return Graph(out)
    if graph_type == "dense":
        # deterministic dense: connect to the next k agents in cyclic order
        out = []
        for i in range(num_agents):
            neigh = [((i + d) % num_agents) for d in range(1, neighbors + 1)]
            out.append(neigh)
        return Graph(out)
    if graph_type == "grid":
        side = int(round(num_agents ** 0.5))
        if side * side != num_agents:
            raise ValueError("grid graph requires num_agents to be a perfect square")
        out = []
        for idx in range(num_agents):
            r, c = divmod(idx, side)
            neigh = []
            if r > 0:
                neigh.append((r - 1) * side + c)
            if r < side - 1:
                neigh.append((r + 1) * side + c)
            if c > 0:
                neigh.append(r * side + (c - 1))
            if c < side - 1:
                neigh.append(r * side + (c + 1))
            out.append(neigh)
        return Graph(out)
    if graph_type == "star":
        out = []
        for i in range(num_agents):
            if i == 0:
                out.append([j for j in range(1, num_agents)])
            else:
                out.append([0])
        return Graph(out)
    raise ValueError(f"Unknown graph_type: {graph_type}")



def sample_observation_masks(num_agents: int, obs_dim: int, observer_frac: float, min_obs_dims: int) -> Tuple[np.ndarray, np.ndarray]:
    num_observers = max(1, int(round(num_agents * observer_frac)))
    observer_idx = set(random.sample(range(num_agents), k=num_observers))

    masks = np.zeros((num_agents, obs_dim), dtype=np.float32)
    for i in range(num_agents):
        if i in observer_idx:
            # Random mask with at least min_obs_dims
            while True:
                m = np.random.rand(obs_dim) < 0.5
                if m.sum() >= min_obs_dims:
                    masks[i] = m.astype(np.float32)
                    break
        else:
            masks[i] = 0.0

    is_observer = masks.sum(axis=1) > 0
    # Ensure at least one non-observer for evaluation split
    if is_observer.all():
        # force one agent to be non-observer
        idx = random.randrange(num_agents)
        masks[idx] = 0.0
        is_observer[idx] = False
    return masks, is_observer



def env_reset(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out


def env_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        return obs, reward, done, info
    obs, reward, done, info = out
    return obs, reward, done, info


def get_action_dim(space) -> int:
    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    if isinstance(space, gym.spaces.Box):
        return int(np.prod(space.shape))
    if isinstance(space, gym.spaces.MultiDiscrete):
        return int(np.sum(space.nvec))
    if isinstance(space, gym.spaces.MultiBinary):
        return int(space.n)
    raise ValueError(f"Unsupported action space: {space}")


def action_to_vector(action, space) -> np.ndarray:
    if isinstance(space, gym.spaces.Discrete):
        vec = np.zeros(space.n, dtype=np.float32)
        vec[int(action)] = 1.0
        return vec
    if isinstance(space, gym.spaces.Box):
        return np.asarray(action, dtype=np.float32).ravel()
    if isinstance(space, gym.spaces.MultiDiscrete):
        parts = []
        action_arr = np.asarray(action, dtype=np.int64).ravel()
        for a, n in zip(action_arr, space.nvec):
            v = np.zeros(int(n), dtype=np.float32)
            v[int(a)] = 1.0
            parts.append(v)
        return np.concatenate(parts, axis=0)
    if isinstance(space, gym.spaces.MultiBinary):
        return np.asarray(action, dtype=np.float32).ravel()
    raise ValueError(f"Unsupported action space: {space}")


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--agents", type=int, default=16)
    parser.add_argument("--graph", type=str, default="ring", choices=["ring", "ring-k", "line", "full", "random", "dense", "grid", "star"])
    parser.add_argument("--neighbors", type=int, default=None)
    parser.add_argument("--degree", type=int, default=4)  # deprecated alias for neighbors
    parser.add_argument("--observer-frac", type=float, default=0.5)
    parser.add_argument("--min-obs-dims", type=int, default=1)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps-per-episode", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    set_seed(args.seed)

    env = gym.make(args.env)
    obs = env_reset(env)
    obs = np.asarray(obs, dtype=np.float32).ravel()
    obs_dim = obs.shape[0]
    action_dim = get_action_dim(env.action_space)

    k_neighbors = args.neighbors if args.neighbors is not None else args.degree
    graph = build_graph(args.agents, args.graph, k_neighbors)

    masks, is_observer = sample_observation_masks(
        num_agents=args.agents,
        obs_dim=obs_dim,
        observer_frac=args.observer_frac,
        min_obs_dims=args.min_obs_dims,
    )

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu, --device mps, or --device auto.")
    if args.device == "mps" and not mps_available:
        raise RuntimeError("MPS requested but not available. Use --device cpu, --device cuda, or --device auto.")
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif mps_available:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    agents = nn.ModuleList(
        [Agent(
            obs_dim,
            args.state_dim,
            num_neighbors=len(graph.neighbors[i]),
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
        )
         for i in range(args.agents)]
    ).to(device)

    opt = torch.optim.Adam(agents.parameters(), lr=args.lr)

    print("Env:", args.env)
    print("Obs dim:", obs_dim)
    print("Action dim:", action_dim)
    print("Agents:", args.agents)
    print("Graph:", args.graph)
    if args.graph not in {"full", "ring", "line", "grid", "star"}:
        print("Neighbors per agent:", k_neighbors)
    print("Hidden dim:", args.hidden_dim)
    print("Hidden layers:", args.hidden_layers)
    print("Device:", device)
    print("Observer agents:", int(is_observer.sum()))
    print("Non-observer agents:", int((~is_observer).sum()))

    for ep in range(1, args.episodes + 1):
        obs = env_reset(env)
        obs = np.asarray(obs, dtype=np.float32).ravel()

        # running metrics
        total_mse = 0.0
        total_count = 0
        obs_mse = 0.0
        obs_count = 0
        nonobs_mse = 0.0
        nonobs_count = 0

        for _ in range(args.steps_per_episode):
            # encode current state for all agents
            obs_stack = np.stack([obs] * args.agents, axis=0)
            masked_obs = obs_stack * masks
            masked_obs_t = to_tensor(masked_obs, device)

            s_t = []
            for i in range(args.agents):
                s_t.append(agents[i].encode(masked_obs_t[i:i+1]))
            s_t = torch.cat(s_t, dim=0)  # [A, state_dim]

            # take action (and pass to observers only)
            action = env.action_space.sample()
            action_vec = action_to_vector(action, env.action_space)
            action_vec_t = to_tensor(action_vec[None, :], device)
            action_zero_t = torch.zeros_like(action_vec_t)
            obs_next, _, done, _ = env_step(env, action)
            obs_next = np.asarray(obs_next, dtype=np.float32).ravel()

            # encode next state (targets, stop-grad)
            obs_next_stack = np.stack([obs_next] * args.agents, axis=0)
            masked_obs_next = obs_next_stack * masks
            with torch.no_grad():
                masked_obs_next_t = to_tensor(masked_obs_next, device)
                s_next = []
                for i in range(args.agents):
                    s_next.append(agents[i].encode(masked_obs_next_t[i:i+1]))
                s_next = torch.cat(s_next, dim=0)

            loss = 0.0
            step_mse = []
            for i in range(args.agents):
                neigh = graph.neighbors[i]
                neigh_states = s_t[neigh].unsqueeze(0)  # [1, N, state_dim]
                action_in = action_vec_t if is_observer[i] else action_zero_t
                pred = agents[i].predict_neighbors(s_t[i:i+1], neigh_states, action_in)  # [1, N, state_dim]
                target = s_next[neigh].unsqueeze(0)
                mse = F.mse_loss(pred, target, reduction="mean")
                loss = loss + mse
                step_mse.append(mse.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

            # metrics split by observer vs non-observer agents
            for i, mse_val in enumerate(step_mse):
                total_mse += mse_val
                total_count += 1
                if is_observer[i]:
                    obs_mse += mse_val
                    obs_count += 1
                else:
                    nonobs_mse += mse_val
                    nonobs_count += 1

            obs = obs_next
            if done:
                obs = env_reset(env)
                obs = np.asarray(obs, dtype=np.float32).ravel()

        if ep % args.log_interval == 0:
            total_avg = total_mse / max(1, total_count)
            obs_avg = obs_mse / max(1, obs_count)
            nonobs_avg = nonobs_mse / max(1, nonobs_count)
            print(f"Ep {ep:4d} | MSE observers: {obs_avg:.6f} | MSE non-observers: {nonobs_avg:.6f} | MSE total: {total_avg:.6f}")

    env.close()


if __name__ == "__main__":
    main()
