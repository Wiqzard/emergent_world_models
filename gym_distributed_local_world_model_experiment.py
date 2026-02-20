import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import wandb
except Exception:
    wandb = None

try:
    import gymnasium as gym
except Exception:
    import gym


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if device_arg == "mps" and not mps_available:
        raise RuntimeError("MPS requested but not available.")
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def env_reset(env, seed: Optional[int] = None) -> np.ndarray:
    if seed is None:
        out = env.reset()
    else:
        out = env.reset(seed=seed)
    if isinstance(out, tuple) and len(out) == 2:
        obs = out[0]
    else:
        obs = out
    return np.asarray(obs, dtype=np.float32).ravel()


def env_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = out
        done = bool(done)
    return np.asarray(obs, dtype=np.float32).ravel(), reward, done, info


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


@dataclass
class Graph:
    neighbor_idx: torch.Tensor
    neighbor_mask: torch.Tensor
    edge_feat: torch.Tensor


def build_graph(num_agents: int, graph_type: str, device: torch.device) -> Graph:
    if graph_type not in {"ring", "line"}:
        raise ValueError("Only ring and line graphs are supported.")

    max_neighbors = 2
    neighbor_idx = torch.full((num_agents, max_neighbors), -1, dtype=torch.long)
    neighbor_mask = torch.zeros((num_agents, max_neighbors), dtype=torch.float32)
    edge_feat = torch.zeros((num_agents, max_neighbors, 1), dtype=torch.float32)

    for i in range(num_agents):
        if graph_type == "ring":
            left = (i - 1) % num_agents
            right = (i + 1) % num_agents
            neighbor_idx[i, 0] = left
            neighbor_idx[i, 1] = right
            neighbor_mask[i, 0] = 1.0
            neighbor_mask[i, 1] = 1.0
            edge_feat[i, 0, 0] = -1.0
            edge_feat[i, 1, 0] = 1.0
        else:
            if i - 1 >= 0:
                neighbor_idx[i, 0] = i - 1
                neighbor_mask[i, 0] = 1.0
                edge_feat[i, 0, 0] = -1.0
            if i + 1 < num_agents:
                neighbor_idx[i, 1] = i + 1
                neighbor_mask[i, 1] = 1.0
                edge_feat[i, 1, 0] = 1.0

    return Graph(
        neighbor_idx=neighbor_idx.to(device),
        neighbor_mask=neighbor_mask.to(device),
        edge_feat=edge_feat.to(device),
    )


def sample_observer_mask(num_agents: int, observer_frac: float, seed: int, device: torch.device) -> torch.Tensor:
    observer_frac = float(np.clip(observer_frac, 0.0, 1.0))
    num_observers = int(round(num_agents * observer_frac))
    if observer_frac > 0.0 and num_observers == 0:
        num_observers = 1
    num_observers = min(num_agents, num_observers)

    mask = np.zeros(num_agents, dtype=np.float32)
    if num_observers > 0:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(num_agents, size=num_observers, replace=False)
        mask[chosen] = 1.0
    return torch.as_tensor(mask, dtype=torch.float32, device=device)


def sample_state_part_masks(
    num_agents: int,
    obs_dim: int,
    observer_mask: torch.Tensor,
    min_obs_dims: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if min_obs_dims < 1:
        raise ValueError("--min-obs-dims must be >= 1")
    min_obs_dims = min(min_obs_dims, obs_dim)

    rng = np.random.default_rng(seed + 101)
    masks = np.zeros((num_agents, obs_dim), dtype=np.float32)
    observer_np = observer_mask.detach().cpu().numpy() > 0.5

    for i in range(num_agents):
        if not observer_np[i]:
            continue
        chosen = rng.choice(obs_dim, size=min_obs_dims, replace=False)
        masks[i, chosen] = 1.0
        if min_obs_dims < obs_dim:
            extra_keep = rng.random(obs_dim) < 0.2
            masks[i] = np.maximum(masks[i], extra_keep.astype(np.float32))

    return torch.as_tensor(masks, dtype=torch.float32, device=device)


def sample_identity_assignments(
    batch_size: int,
    num_agents: int,
    permute_agent_positions: bool,
    device: torch.device,
) -> torch.Tensor:
    if not permute_agent_positions:
        idx = np.tile(np.arange(num_agents, dtype=np.int64), (batch_size, 1))
    else:
        idx = np.stack(
            [np.random.permutation(num_agents).astype(np.int64) for _ in range(batch_size)],
            axis=0,
        )
    return torch.as_tensor(idx, dtype=torch.long, device=device)


class GymBatchRollout:
    def __init__(
        self,
        env_name: str,
        batch_size: int,
        seed: int,
        render_mode: Optional[str] = None,
        frame_skip: int = 1,
    ):
        self.env_name = env_name
        self.batch_size = batch_size
        self.base_seed = seed
        self.render_mode = render_mode
        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")
        self.frame_skip = int(frame_skip)
        if render_mode is None:
            self.envs = [gym.make(env_name) for _ in range(batch_size)]
        else:
            try:
                self.envs = [gym.make(env_name, render_mode=render_mode) for _ in range(batch_size)]
            except TypeError as exc:
                raise RuntimeError(
                    f"Environment {env_name} does not accept render_mode={render_mode}."
                ) from exc
        self.step_count = 0

        obs0 = env_reset(self.envs[0], seed=seed)
        self.obs_dim = int(obs0.shape[0])
        self.action_space = self.envs[0].action_space
        self.action_dim = get_action_dim(self.action_space)

        self.current_obs = []
        for i, env in enumerate(self.envs):
            self.current_obs.append(env_reset(env, seed=seed + i))
        self.current_obs = np.stack(self.current_obs, axis=0).astype(np.float32)

    def sample_rollout(
        self,
        seq_len_plus_one: int,
        include_frames: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        states = [self.current_obs.copy()]
        actions = []
        frames = [] if include_frames else None

        for _ in range(seq_len_plus_one - 1):
            action_vecs = np.zeros((self.batch_size, self.action_dim), dtype=np.float32)
            next_obs = np.zeros((self.batch_size, self.obs_dim), dtype=np.float32)
            next_frames = [] if include_frames else None

            for i, env in enumerate(self.envs):
                action_acc = np.zeros((self.action_dim,), dtype=np.float32)
                obs_next = self.current_obs[i]

                for _ in range(self.frame_skip):
                    action = env.action_space.sample()
                    action_acc += action_to_vector(action, env.action_space)
                    obs_next, _, done, _ = env_step(env, action)
                    if done:
                        reset_seed = self.base_seed + 100000 + self.step_count * self.batch_size + i
                        obs_next = env_reset(env, seed=reset_seed)

                # Keep action input compatible: same dimensionality, aggregated over executed actions.
                action_vecs[i] = action_acc / float(self.frame_skip)
                next_obs[i] = obs_next
                if include_frames:
                    try:
                        frame = env.render()
                    except Exception as exc:
                        raise RuntimeError(
                            "Pixel probe rendering failed. Install rendering dependencies "
                            "(for classic-control: `pygame`) or use an env with rgb_array support."
                        ) from exc
                    if frame is None:
                        raise RuntimeError(
                            "Pixel probe requested but env.render() returned None. "
                            "Use an environment that supports rgb_array rendering."
                        )
                    next_frames.append(frame)

            self.step_count += 1
            self.current_obs = next_obs
            actions.append(action_vecs)
            states.append(next_obs.copy())
            if include_frames and frames is not None and next_frames is not None:
                frames.append(np.stack(next_frames, axis=0))

        frames_out = None
        if include_frames and frames is not None:
            frames_out = np.stack(frames, axis=1)
        return np.stack(states, axis=1), np.stack(actions, axis=1), frames_out

    def close(self) -> None:
        for env in self.envs:
            env.close()


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, hidden_layers: int):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(nn.ReLU())
            last = hidden_dim
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DistributedGymWorldModel(nn.Module):
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        id_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        edge_feat_dim: int,
        max_neighbors: int,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.id_dim = id_dim
        self.max_neighbors = max_neighbors

        self.identity_embed = nn.Embedding(num_agents, id_dim)
        self.id_to_latent = nn.Linear(id_dim, latent_dim)

        self.obs_encoder = MLP(obs_dim, latent_dim, hidden_dim, hidden_layers)
        self.action_encoder = MLP(action_dim, latent_dim, hidden_dim, max(1, hidden_layers - 1))
        self.msg_encoder = MLP(latent_dim + 2 * id_dim + edge_feat_dim, latent_dim, hidden_dim, max(1, hidden_layers - 1))

        update_in_dim = latent_dim + latent_dim + latent_dim + id_dim + 1
        self.update_cell = nn.GRUCell(update_in_dim, latent_dim)

        self.self_head = MLP(latent_dim, obs_dim, hidden_dim, hidden_layers)
        self.neighbor_head = MLP(latent_dim, max_neighbors * latent_dim, hidden_dim, hidden_layers)

    def identity_features(self, identity_at_pos: torch.Tensor) -> torch.Tensor:
        return self.identity_embed(identity_at_pos)

    def initial_latent(self, id_features: torch.Tensor) -> torch.Tensor:
        b, a, _ = id_features.shape
        z0 = self.id_to_latent(id_features.reshape(b * a, self.id_dim))
        return z0.reshape(b, a, self.latent_dim)

    def aggregate_messages(
        self,
        z_prev: torch.Tensor,
        id_features: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
        edge_feat: torch.Tensor,
        disable_messages: bool,
    ) -> torch.Tensor:
        if disable_messages:
            return torch.zeros_like(z_prev)

        b, a, _ = z_prev.shape
        idx_safe = neighbor_idx.clamp(min=0)
        recv_ids = id_features
        agg = torch.zeros_like(z_prev)

        for slot in range(idx_safe.shape[1]):
            send_z = z_prev[:, idx_safe[:, slot], :]
            send_ids = id_features[:, idx_safe[:, slot], :]
            rel = edge_feat[:, slot, :].view(1, a, -1).expand(b, a, -1)
            msg_in = torch.cat([send_z, send_ids, recv_ids, rel], dim=-1)
            msg = self.msg_encoder(msg_in.reshape(b * a, -1)).reshape(b, a, self.latent_dim)
            slot_mask = neighbor_mask[:, slot].view(1, a, 1)
            agg = agg + msg * slot_mask

        denom = neighbor_mask.sum(dim=1).clamp(min=1.0).view(1, a, 1)
        return agg / denom

    def step(
        self,
        z_prev: torch.Tensor,
        obs_local: torch.Tensor,
        action_local: torch.Tensor,
        obs_mask: torch.Tensor,
        id_features: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
        edge_feat: torch.Tensor,
        disable_messages: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, a, obs_dim = obs_local.shape
        obs_embed = self.obs_encoder(obs_local.reshape(b * a, obs_dim)).reshape(b, a, self.latent_dim)

        act_dim = action_local.shape[-1]
        act_embed = self.action_encoder(action_local.reshape(b * a, act_dim)).reshape(b, a, self.latent_dim)
        obs_embed = obs_embed * obs_mask.unsqueeze(-1)
        act_embed = act_embed * obs_mask.unsqueeze(-1)

        msg = self.aggregate_messages(
            z_prev=z_prev,
            id_features=id_features,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            edge_feat=edge_feat,
            disable_messages=disable_messages,
        )

        update_in = torch.cat([obs_embed, act_embed, msg, id_features, obs_mask.unsqueeze(-1)], dim=-1)
        z_next = self.update_cell(
            update_in.reshape(b * a, -1),
            z_prev.reshape(b * a, self.latent_dim),
        ).reshape(b, a, self.latent_dim)

        self_pred = self.self_head(z_next.reshape(b * a, self.latent_dim)).reshape(b, a, obs_dim)
        neighbor_pred = self.neighbor_head(z_next.reshape(b * a, self.latent_dim)).reshape(
            b, a, self.max_neighbors, self.latent_dim
        )
        return z_next, self_pred, neighbor_pred


def gather_neighbor_latents(z: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    idx_safe = neighbor_idx.clamp(min=0)
    gathered = []
    for slot in range(idx_safe.shape[1]):
        gathered.append(z[:, idx_safe[:, slot], :].unsqueeze(2))
    return torch.cat(gathered, dim=2)


def compute_sequence_loss(
    model: DistributedGymWorldModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    base_observer_mask: torch.Tensor,
    base_state_mask: torch.Tensor,
    graph: Graph,
    lambda_self: float,
    lambda_neighbor: float,
    blind_reg_weight: float,
    disable_messages: bool,
    permute_agent_positions: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, seq_len_plus_one, obs_dim = states.shape
    num_agents = base_observer_mask.shape[0]
    steps = seq_len_plus_one - 1

    identity_at_pos = sample_identity_assignments(
        batch_size=b,
        num_agents=num_agents,
        permute_agent_positions=permute_agent_positions,
        device=states.device,
    )
    obs_mask = base_observer_mask[identity_at_pos]  # [B, A]
    state_mask = base_state_mask[identity_at_pos]  # [B, A, obs_dim]
    id_features = model.identity_features(identity_at_pos)
    z = model.initial_latent(id_features)

    total_self = torch.zeros((), device=states.device)
    total_neighbor = torch.zeros((), device=states.device)
    total_blind_reg = torch.zeros((), device=states.device)

    neighbor_denom = graph.neighbor_mask.sum().clamp(min=1.0) * b

    for t in range(steps):
        state_t = states[:, t]
        action_t = actions[:, t]
        state_next = states[:, t + 1]

        state_local = state_t.unsqueeze(1).expand(b, num_agents, obs_dim)
        action_local = action_t.unsqueeze(1).expand(b, num_agents, action_t.shape[-1])

        state_local = state_local * state_mask
        action_local = action_local * obs_mask.unsqueeze(-1)

        z_next, self_pred, neighbor_pred = model.step(
            z_prev=z,
            obs_local=state_local,
            action_local=action_local,
            obs_mask=obs_mask,
            id_features=id_features,
            neighbor_idx=graph.neighbor_idx,
            neighbor_mask=graph.neighbor_mask,
            edge_feat=graph.edge_feat,
            disable_messages=disable_messages,
        )

        target_local_next = state_next.unsqueeze(1).expand(b, num_agents, obs_dim)
        self_mask = state_mask * obs_mask.unsqueeze(-1)
        self_sq = (self_pred - target_local_next) ** 2
        self_denom = self_mask.sum().clamp(min=1.0)
        loss_self = (self_sq * self_mask).sum() / self_denom

        target_neighbors = gather_neighbor_latents(z_next.detach(), graph.neighbor_idx)
        neighbor_sq = ((neighbor_pred - target_neighbors) ** 2).mean(dim=-1)
        loss_neighbor = (neighbor_sq * graph.neighbor_mask.view(1, num_agents, -1)).sum() / neighbor_denom

        blind_mask = 1.0 - obs_mask
        blind_denom = blind_mask.sum().clamp(min=1.0)
        loss_blind_reg = ((z_next ** 2).mean(dim=-1) * blind_mask).sum() / blind_denom

        total_self = total_self + loss_self
        total_neighbor = total_neighbor + loss_neighbor
        total_blind_reg = total_blind_reg + loss_blind_reg
        z = z_next

    mean_self = total_self / steps
    mean_neighbor = total_neighbor / steps
    mean_blind_reg = total_blind_reg / steps
    total = lambda_self * mean_self + lambda_neighbor * mean_neighbor + blind_reg_weight * mean_blind_reg
    return total, mean_self, mean_neighbor, mean_blind_reg


@torch.no_grad()
def evaluate_model(
    model: DistributedGymWorldModel,
    rollout: GymBatchRollout,
    eval_batches: int,
    seq_len: int,
    device: torch.device,
    base_observer_mask: torch.Tensor,
    base_state_mask: torch.Tensor,
    graph: Graph,
    lambda_self: float,
    lambda_neighbor: float,
    blind_reg_weight: float,
    disable_messages: bool,
    permute_agent_positions: bool,
) -> Dict[str, float]:
    model.eval()
    total = 0.0
    self_total = 0.0
    neighbor_total = 0.0
    blind_total = 0.0

    for _ in range(eval_batches):
        states_np, actions_np, _ = rollout.sample_rollout(seq_len_plus_one=seq_len + 1)
        states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)
        loss, self_loss, neighbor_loss, blind_loss = compute_sequence_loss(
            model=model,
            states=states,
            actions=actions,
            base_observer_mask=base_observer_mask,
            base_state_mask=base_state_mask,
            graph=graph,
            lambda_self=lambda_self,
            lambda_neighbor=lambda_neighbor,
            blind_reg_weight=blind_reg_weight,
            disable_messages=disable_messages,
            permute_agent_positions=permute_agent_positions,
        )
        total += float(loss.item())
        self_total += float(self_loss.item())
        neighbor_total += float(neighbor_loss.item())
        blind_total += float(blind_loss.item())

    denom = float(max(1, eval_batches))
    return {
        "loss": total / denom,
        "self_loss": self_total / denom,
        "neighbor_loss": neighbor_total / denom,
        "blind_reg_loss": blind_total / denom,
    }


@torch.no_grad()
def collect_probe_dataset(
    model: DistributedGymWorldModel,
    rollout: GymBatchRollout,
    num_batches: int,
    seq_len: int,
    device: torch.device,
    base_observer_mask: torch.Tensor,
    base_state_mask: torch.Tensor,
    graph: Graph,
    disable_messages: bool,
    permute_agent_positions: bool,
    include_agent_latents: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    model.eval()
    global_x = []
    agent_x = []
    targets = []

    num_agents = base_observer_mask.shape[0]
    obs_dim = base_state_mask.shape[1]

    for _ in range(num_batches):
        states_np, actions_np, _ = rollout.sample_rollout(seq_len_plus_one=seq_len + 1)
        states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)
        b = states.shape[0]

        identity_at_pos = sample_identity_assignments(
            batch_size=b,
            num_agents=num_agents,
            permute_agent_positions=permute_agent_positions,
            device=device,
        )
        obs_mask = base_observer_mask[identity_at_pos]
        state_mask = base_state_mask[identity_at_pos]
        id_features = model.identity_features(identity_at_pos)
        z = model.initial_latent(id_features)

        for t in range(seq_len):
            state_t = states[:, t]
            action_t = actions[:, t]
            state_local = state_t.unsqueeze(1).expand(b, num_agents, obs_dim) * state_mask
            action_local = action_t.unsqueeze(1).expand(b, num_agents, action_t.shape[-1]) * obs_mask.unsqueeze(-1)
            z, _, _ = model.step(
                z_prev=z,
                obs_local=state_local,
                action_local=action_local,
                obs_mask=obs_mask,
                id_features=id_features,
                neighbor_idx=graph.neighbor_idx,
                neighbor_mask=graph.neighbor_mask,
                edge_feat=graph.edge_feat,
                disable_messages=disable_messages,
            )
            global_x.append(z.reshape(b, -1).cpu())
            if include_agent_latents:
                agent_x.append(z.cpu())
            targets.append(states[:, t + 1].cpu())

    x_global = torch.cat(global_x, dim=0)
    y = torch.cat(targets, dim=0)
    x_agent = torch.cat(agent_x, dim=0) if include_agent_latents else None
    return x_global, x_agent, y


def preprocess_frames_to_vectors(
    frames_np: np.ndarray,
    pixel_height: int,
    pixel_width: int,
    device: torch.device,
) -> torch.Tensor:
    # frames_np: [B, T, H0, W0, C]
    frames = torch.as_tensor(frames_np, dtype=torch.float32, device=device)
    if frames.max() > 1.0:
        frames = frames / 255.0
    if frames.shape[-1] == 1:
        frames = frames.repeat_interleave(3, dim=-1)
    b, t, h0, w0, c = frames.shape
    frames_chw = frames.permute(0, 1, 4, 2, 3).reshape(b * t, c, h0, w0)
    resized = F.interpolate(frames_chw, size=(pixel_height, pixel_width), mode="bilinear", align_corners=False)
    return resized.reshape(b, t, -1)


@torch.no_grad()
def collect_pixel_probe_dataset(
    model: DistributedGymWorldModel,
    rollout: GymBatchRollout,
    num_batches: int,
    seq_len: int,
    device: torch.device,
    base_observer_mask: torch.Tensor,
    base_state_mask: torch.Tensor,
    graph: Graph,
    disable_messages: bool,
    permute_agent_positions: bool,
    pixel_height: int,
    pixel_width: int,
    pixel_horizon: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    global_x = []
    pixel_targets = []

    num_agents = base_observer_mask.shape[0]
    obs_dim = base_state_mask.shape[1]

    for _ in range(num_batches):
        states_np, actions_np, frames_np = rollout.sample_rollout(
            seq_len_plus_one=seq_len + pixel_horizon,
            include_frames=True,
        )
        if frames_np is None:
            raise RuntimeError("Pixel probe requested but no frames were captured from rollout.")

        states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)
        pixel_vecs = preprocess_frames_to_vectors(frames_np, pixel_height=pixel_height, pixel_width=pixel_width, device=device)
        b = states.shape[0]

        identity_at_pos = sample_identity_assignments(
            batch_size=b,
            num_agents=num_agents,
            permute_agent_positions=permute_agent_positions,
            device=device,
        )
        obs_mask = base_observer_mask[identity_at_pos]
        state_mask = base_state_mask[identity_at_pos]
        id_features = model.identity_features(identity_at_pos)
        z = model.initial_latent(id_features)

        for t in range(seq_len):
            state_t = states[:, t]
            action_t = actions[:, t]
            state_local = state_t.unsqueeze(1).expand(b, num_agents, obs_dim) * state_mask
            action_local = action_t.unsqueeze(1).expand(b, num_agents, action_t.shape[-1]) * obs_mask.unsqueeze(-1)
            z, _, _ = model.step(
                z_prev=z,
                obs_local=state_local,
                action_local=action_local,
                obs_mask=obs_mask,
                id_features=id_features,
                neighbor_idx=graph.neighbor_idx,
                neighbor_mask=graph.neighbor_mask,
                edge_feat=graph.edge_feat,
                disable_messages=disable_messages,
            )
            # Condition pixel probe on the intermediate action window needed for t+K prediction.
            action_window = actions[:, t : t + pixel_horizon].mean(dim=1)
            feat = torch.cat([z.reshape(b, -1), action_window], dim=-1)
            global_x.append(feat.cpu())
            target_idx = t + pixel_horizon - 1
            pixel_targets.append(pixel_vecs[:, target_idx].cpu())

    return torch.cat(global_x, dim=0), torch.cat(pixel_targets, dim=0)


def fit_ridge_regression(x: torch.Tensor, y: torch.Tensor, l2: float) -> torch.Tensor:
    ones = torch.ones((x.shape[0], 1), dtype=x.dtype)
    x_bias = torch.cat([x, ones], dim=1)
    x64 = x_bias.double()
    y64 = y.double()
    xtx = x64.T @ x64
    reg = torch.eye(xtx.shape[0], dtype=torch.float64) * float(l2)
    reg[-1, -1] = 0.0
    try:
        w = torch.linalg.solve(xtx + reg, x64.T @ y64)
    except RuntimeError:
        w = torch.linalg.pinv(xtx + reg) @ (x64.T @ y64)
    return w.float()


def ridge_predict(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((x.shape[0], 1), dtype=x.dtype)
    x_bias = torch.cat([x, ones], dim=1)
    return x_bias @ w


def mse_value(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred - target) ** 2).item())


def evaluate_agent_probes(
    x_agent_train: torch.Tensor,
    x_agent_test: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    observer_mask: torch.Tensor,
    l2: float,
) -> Dict[str, float]:
    observer_np = observer_mask.cpu().numpy() > 0.5
    observer_mse = []
    blind_mse = []
    for idx in range(x_agent_train.shape[1]):
        w = fit_ridge_regression(x_agent_train[:, idx, :], y_train, l2)
        pred = ridge_predict(x_agent_test[:, idx, :], w)
        mse = mse_value(pred, y_test)
        if observer_np[idx]:
            observer_mse.append(mse)
        else:
            blind_mse.append(mse)
    return {
        "observer_agent_probe_mse": float(np.mean(observer_mse)) if observer_mse else float("nan"),
        "blind_agent_probe_mse": float(np.mean(blind_mse)) if blind_mse else float("nan"),
    }


def create_pixel_prediction_plot(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    pixel_height: int,
    pixel_width: int,
    output_path: str,
    max_frames: int = 6,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not installed; cannot create pixel prediction plot.")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n = min(max_frames, y_true.shape[0])
    true_img = y_true[:n].reshape(n, 3, pixel_height, pixel_width).permute(0, 2, 3, 1).numpy()
    pred_img = y_pred[:n].reshape(n, 3, pixel_height, pixel_width).permute(0, 2, 3, 1).numpy()
    true_img = np.clip(true_img, 0.0, 1.0)
    pred_img = np.clip(pred_img, 0.0, 1.0)

    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.2))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col in range(n):
        axes[0, col].imshow(true_img[col])
        axes[0, col].set_title(f"True t+1 #{col+1}", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(pred_img[col])
        axes[1, col].set_title(f"Pred t+1 #{col+1}", fontsize=9)
        axes[1, col].axis("off")

    fig.suptitle("Global Pixel Prediction From Distributed Latent", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def create_metrics_plot(
    history: Dict[str, List[float]],
    final_metrics: Dict[str, float],
    output_path: str,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not installed; cannot create metrics plot.")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    epochs = np.array(history["epoch"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train total")
    axes[0].plot(epochs, history["eval_loss"], label="eval total")
    axes[0].plot(epochs, history["train_self"], label="train self", alpha=0.7)
    axes[0].plot(epochs, history["train_nbr"], label="train nbr", alpha=0.7)
    axes[0].set_title("Training Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)

    axes[1].bar(
        ["Global test MSE", "Baseline MSE"],
        [final_metrics["global_test_mse"], final_metrics["baseline_test_mse"]],
    )
    axes[1].set_title("Global Probe")
    axes[1].set_ylabel("MSE")
    axes[1].tick_params(axis="x", rotation=20)

    obs_mse = final_metrics.get("observer_agent_probe_mse", np.nan)
    blind_mse = final_metrics.get("blind_agent_probe_mse", np.nan)
    axes[2].bar(["Observer agent", "Blind agent"], [obs_mse, blind_mse])
    axes[2].set_title("Single-Agent Probe")
    axes[2].set_ylabel("MSE")
    axes[2].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gym distributed world model: observers get local state parts + random actions; evaluate global emergence."
    )
    parser.add_argument("--env", type=str, default="Acrobot-v1")
    parser.add_argument("--agents", type=int, default=32)
    parser.add_argument("--graph", type=str, default="ring", choices=["ring", "line"])
    parser.add_argument("--observer-frac", type=float, default=0.5)
    parser.add_argument("--min-obs-dims", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--id-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=40)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-self", type=float, default=1.0)
    parser.add_argument("--lambda-neighbor", type=float, default=1.0)
    parser.add_argument("--blind-reg-weight", type=float, default=1e-3)
    parser.add_argument("--disable-messages", action="store_true")
    parser.add_argument("--permute-agent-positions", action="store_true")
    parser.add_argument("--probe-train-batches", type=int, default=24)
    parser.add_argument("--probe-test-batches", type=int, default=12)
    parser.add_argument("--probe-l2", type=float, default=1e-2)
    parser.add_argument("--pixel-probe", action="store_true")
    parser.add_argument("--pixel-probe-train-batches", type=int, default=12)
    parser.add_argument("--pixel-probe-test-batches", type=int, default=6)
    parser.add_argument("--pixel-probe-batch-size", type=int, default=4)
    parser.add_argument("--pixel-height", type=int, default=84)
    parser.add_argument("--pixel-width", type=int, default=84)
    parser.add_argument("--pixel-horizon", type=int, default=1)
    parser.add_argument("--pixel-probe-l2", type=float, default=1e-2)
    parser.add_argument("--pixel-vis-frames", type=int, default=6)
    parser.add_argument("--pixel-plot-file", type=str, default="outputs/gym_pixel_prediction_comparison.png")
    parser.add_argument("--skip-agent-probes", action="store_true")
    parser.add_argument("--plot-file", type=str, default="outputs/gym_world_model_metrics.png")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="emergent-world-models")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    if args.agents < 2:
        raise ValueError("--agents must be >= 2")
    if args.seq_len < 1:
        raise ValueError("--seq-len must be >= 1")
    if args.frame_skip < 1:
        raise ValueError("--frame-skip must be >= 1")
    if args.pixel_horizon < 1:
        raise ValueError("--pixel-horizon must be >= 1")

    set_seed(args.seed)
    device = resolve_device(args.device)

    train_rollout = GymBatchRollout(args.env, args.batch_size, seed=args.seed + 11, frame_skip=args.frame_skip)
    eval_rollout = GymBatchRollout(args.env, args.batch_size, seed=args.seed + 17, frame_skip=args.frame_skip)
    probe_train_rollout = GymBatchRollout(args.env, args.batch_size, seed=args.seed + 23, frame_skip=args.frame_skip)
    probe_test_rollout = GymBatchRollout(args.env, args.batch_size, seed=args.seed + 29, frame_skip=args.frame_skip)
    pixel_probe_train_rollout = None
    pixel_probe_test_rollout = None
    if args.pixel_probe:
        pixel_probe_train_rollout = GymBatchRollout(
            args.env,
            args.pixel_probe_batch_size,
            seed=args.seed + 31,
            render_mode="rgb_array",
            frame_skip=args.frame_skip,
        )
        pixel_probe_test_rollout = GymBatchRollout(
            args.env,
            args.pixel_probe_batch_size,
            seed=args.seed + 37,
            render_mode="rgb_array",
            frame_skip=args.frame_skip,
        )

    obs_dim = train_rollout.obs_dim
    action_dim = train_rollout.action_dim

    base_observer_mask = sample_observer_mask(args.agents, args.observer_frac, args.seed, device)
    base_state_mask = sample_state_part_masks(
        num_agents=args.agents,
        obs_dim=obs_dim,
        observer_mask=base_observer_mask,
        min_obs_dims=args.min_obs_dims,
        seed=args.seed,
        device=device,
    )
    graph = build_graph(args.agents, args.graph, device=device)

    model = DistributedGymWorldModel(
        num_agents=args.agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        id_dim=args.id_dim,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        edge_feat_dim=graph.edge_feat.shape[-1],
        max_neighbors=graph.neighbor_idx.shape[1],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_observers = int((base_observer_mask > 0.5).sum().item())
    num_blind = args.agents - num_observers

    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb requested but package is not installed.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    print("Experiment: gym_distributed_local_world_model_experiment.py")
    print(f"Env: {args.env}")
    print(f"Device: {device}")
    print(f"Obs dim: {obs_dim} | Action dim: {action_dim}")
    print(f"Agents: {args.agents} | Graph: {args.graph}")
    print(f"Observers: {num_observers} | Blind: {num_blind}")
    print(f"Permute identity-position: {args.permute_agent_positions}")
    print(f"Disable messages: {args.disable_messages}")
    print(f"Pixel probe enabled: {args.pixel_probe}")
    print(f"Frame skip (macro-step): {args.frame_skip}")
    if args.pixel_probe:
        print(f"Pixel probe horizon (macro-steps ahead): {args.pixel_horizon}")
    print(
        "Loss weights: "
        f"self={args.lambda_self}, "
        f"neighbor={args.lambda_neighbor}, "
        f"blind_reg={args.blind_reg_weight}"
    )

    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_self": [],
        "train_nbr": [],
        "eval_loss": [],
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        train_self = 0.0
        train_nbr = 0.0
        train_blind = 0.0

        for _ in range(args.steps_per_epoch):
            states_np, actions_np, _ = train_rollout.sample_rollout(seq_len_plus_one=args.seq_len + 1)
            states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
            actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)

            loss, self_loss, nbr_loss, blind_loss = compute_sequence_loss(
                model=model,
                states=states,
                actions=actions,
                base_observer_mask=base_observer_mask,
                base_state_mask=base_state_mask,
                graph=graph,
                lambda_self=args.lambda_self,
                lambda_neighbor=args.lambda_neighbor,
                blind_reg_weight=args.blind_reg_weight,
                disable_messages=args.disable_messages,
                permute_agent_positions=args.permute_agent_positions,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total += float(loss.item())
            train_self += float(self_loss.item())
            train_nbr += float(nbr_loss.item())
            train_blind += float(blind_loss.item())

        metrics = evaluate_model(
            model=model,
            rollout=eval_rollout,
            eval_batches=args.eval_batches,
            seq_len=args.seq_len,
            device=device,
            base_observer_mask=base_observer_mask,
            base_state_mask=base_state_mask,
            graph=graph,
            lambda_self=args.lambda_self,
            lambda_neighbor=args.lambda_neighbor,
            blind_reg_weight=args.blind_reg_weight,
            disable_messages=args.disable_messages,
            permute_agent_positions=args.permute_agent_positions,
        )
        denom = float(args.steps_per_epoch)
        train_total /= denom
        train_self /= denom
        train_nbr /= denom
        train_blind /= denom

        history["epoch"].append(epoch)
        history["train_loss"].append(train_total)
        history["train_self"].append(train_self)
        history["train_nbr"].append(train_nbr)
        history["eval_loss"].append(metrics["loss"])

        if epoch % args.log_interval == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"Epoch {epoch:4d} | "
                f"train={train_total:.6f} (self={train_self:.6f}, nbr={train_nbr:.6f}, blind={train_blind:.6f}) | "
                f"eval={metrics['loss']:.6f} (self={metrics['self_loss']:.6f}, nbr={metrics['neighbor_loss']:.6f}, blind={metrics['blind_reg_loss']:.6f})"
            )

        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_total,
                    "train/self_loss": train_self,
                    "train/neighbor_loss": train_nbr,
                    "train/blind_reg_loss": train_blind,
                    "eval/loss": metrics["loss"],
                    "eval/self_loss": metrics["self_loss"],
                    "eval/neighbor_loss": metrics["neighbor_loss"],
                    "eval/blind_reg_loss": metrics["blind_reg_loss"],
                }
            )

    print("Collecting latent datasets for global probe...")
    x_global_train, x_agent_train, y_train = collect_probe_dataset(
        model=model,
        rollout=probe_train_rollout,
        num_batches=args.probe_train_batches,
        seq_len=args.seq_len,
        device=device,
        base_observer_mask=base_observer_mask,
        base_state_mask=base_state_mask,
        graph=graph,
        disable_messages=args.disable_messages,
        permute_agent_positions=args.permute_agent_positions,
        include_agent_latents=not args.skip_agent_probes,
    )
    x_global_test, x_agent_test, y_test = collect_probe_dataset(
        model=model,
        rollout=probe_test_rollout,
        num_batches=args.probe_test_batches,
        seq_len=args.seq_len,
        device=device,
        base_observer_mask=base_observer_mask,
        base_state_mask=base_state_mask,
        graph=graph,
        disable_messages=args.disable_messages,
        permute_agent_positions=args.permute_agent_positions,
        include_agent_latents=not args.skip_agent_probes,
    )

    print("Fitting global linear probe...")
    w_global = fit_ridge_regression(x_global_train, y_train, l2=args.probe_l2)
    y_train_pred = ridge_predict(x_global_train, w_global)
    y_test_pred = ridge_predict(x_global_test, w_global)
    global_train_mse = mse_value(y_train_pred, y_train)
    global_test_mse = mse_value(y_test_pred, y_test)

    baseline_mean = y_train.mean(dim=0, keepdim=True)
    baseline_test_mse = mse_value(baseline_mean.expand_as(y_test), y_test)
    sse = torch.sum((y_test_pred - y_test) ** 2)
    sst = torch.sum((y_test - y_test.mean(dim=0, keepdim=True)) ** 2).clamp(min=1e-8)
    global_r2 = float((1.0 - sse / sst).item())

    final_metrics = {
        "global_train_mse": global_train_mse,
        "global_test_mse": global_test_mse,
        "baseline_test_mse": baseline_test_mse,
        "global_r2": global_r2,
    }

    if not args.skip_agent_probes and x_agent_train is not None and x_agent_test is not None:
        if args.permute_agent_positions:
            print("Per-agent probes skipped due to identity-position permutation.")
        else:
            agent_metrics = evaluate_agent_probes(
                x_agent_train=x_agent_train,
                x_agent_test=x_agent_test,
                y_train=y_train,
                y_test=y_test,
                observer_mask=base_observer_mask,
                l2=args.probe_l2,
            )
            final_metrics.update(agent_metrics)
    print("Global probe results:")
    print(f"- train_mse: {final_metrics['global_train_mse']:.6f}")
    print(f"- test_mse: {final_metrics['global_test_mse']:.6f}")
    print(f"- test_mse_baseline(mean): {final_metrics['baseline_test_mse']:.6f}")
    print(f"- test_r2: {final_metrics['global_r2']:.6f}")
    if "observer_agent_probe_mse" in final_metrics:
        print(f"- observer_agent_probe_mse: {final_metrics['observer_agent_probe_mse']:.6f}")
        print(f"- blind_agent_probe_mse: {final_metrics['blind_agent_probe_mse']:.6f}")

    pixel_plot_saved = False
    if args.pixel_probe:
        if pixel_probe_train_rollout is None or pixel_probe_test_rollout is None:
            raise RuntimeError("Pixel probe enabled but pixel rollouts were not initialized.")

        print("Collecting pixel datasets for global latent->pixel probe...")
        x_pix_train, y_pix_train = collect_pixel_probe_dataset(
            model=model,
            rollout=pixel_probe_train_rollout,
            num_batches=args.pixel_probe_train_batches,
            seq_len=args.seq_len,
            device=device,
            base_observer_mask=base_observer_mask,
            base_state_mask=base_state_mask,
            graph=graph,
            disable_messages=args.disable_messages,
            permute_agent_positions=args.permute_agent_positions,
            pixel_height=args.pixel_height,
            pixel_width=args.pixel_width,
            pixel_horizon=args.pixel_horizon,
        )
        x_pix_test, y_pix_test = collect_pixel_probe_dataset(
            model=model,
            rollout=pixel_probe_test_rollout,
            num_batches=args.pixel_probe_test_batches,
            seq_len=args.seq_len,
            device=device,
            base_observer_mask=base_observer_mask,
            base_state_mask=base_state_mask,
            graph=graph,
            disable_messages=args.disable_messages,
            permute_agent_positions=args.permute_agent_positions,
            pixel_height=args.pixel_height,
            pixel_width=args.pixel_width,
            pixel_horizon=args.pixel_horizon,
        )

        w_pix = fit_ridge_regression(x_pix_train, y_pix_train, l2=args.pixel_probe_l2)
        y_pix_train_pred = ridge_predict(x_pix_train, w_pix)
        y_pix_test_pred = ridge_predict(x_pix_test, w_pix)
        pixel_train_mse = mse_value(y_pix_train_pred, y_pix_train)
        pixel_test_mse = mse_value(y_pix_test_pred, y_pix_test)
        pixel_baseline = y_pix_train.mean(dim=0, keepdim=True)
        pixel_baseline_mse = mse_value(pixel_baseline.expand_as(y_pix_test), y_pix_test)
        pixel_sse = torch.sum((y_pix_test_pred - y_pix_test) ** 2)
        pixel_sst = torch.sum((y_pix_test - y_pix_test.mean(dim=0, keepdim=True)) ** 2).clamp(min=1e-8)
        pixel_r2 = float((1.0 - pixel_sse / pixel_sst).item())

        final_metrics["pixel_train_mse"] = pixel_train_mse
        final_metrics["pixel_test_mse"] = pixel_test_mse
        final_metrics["pixel_baseline_test_mse"] = pixel_baseline_mse
        final_metrics["pixel_test_r2"] = pixel_r2

        print(f"Pixel probe results (global latent -> RGB frame at t+{args.pixel_horizon} macro-steps):")
        print(f"- pixel_train_mse: {pixel_train_mse:.6f}")
        print(f"- pixel_test_mse: {pixel_test_mse:.6f}")
        print(f"- pixel_test_mse_baseline(mean): {pixel_baseline_mse:.6f}")
        print(f"- pixel_test_r2: {pixel_r2:.6f}")

        if plt is None:
            print("matplotlib is not installed; skipping pixel comparison image export.")
        else:
            create_pixel_prediction_plot(
                y_true=y_pix_test,
                y_pred=y_pix_test_pred,
                pixel_height=args.pixel_height,
                pixel_width=args.pixel_width,
                output_path=args.pixel_plot_file,
                max_frames=args.pixel_vis_frames,
            )
            pixel_plot_saved = True
            print(f"Saved pixel comparison: {args.pixel_plot_file}")

    plot_saved = False
    if plt is None:
        print("matplotlib is not installed; skipping plot export.")
    else:
        create_metrics_plot(history=history, final_metrics=final_metrics, output_path=args.plot_file)
        plot_saved = True
        print(f"Saved plot: {args.plot_file}")

    if wandb_run is not None:
        log_payload = {
            "probe/global_train_mse": final_metrics["global_train_mse"],
            "probe/global_test_mse": final_metrics["global_test_mse"],
            "probe/baseline_test_mse": final_metrics["baseline_test_mse"],
            "probe/global_r2": final_metrics["global_r2"],
        }
        if plot_saved:
            log_payload["plot/metrics"] = wandb.Image(args.plot_file)
        if "observer_agent_probe_mse" in final_metrics:
            log_payload["probe/observer_agent_probe_mse"] = final_metrics["observer_agent_probe_mse"]
            log_payload["probe/blind_agent_probe_mse"] = final_metrics["blind_agent_probe_mse"]
        if "pixel_train_mse" in final_metrics:
            log_payload["pixel_probe/train_mse"] = final_metrics["pixel_train_mse"]
            log_payload["pixel_probe/test_mse"] = final_metrics["pixel_test_mse"]
            log_payload["pixel_probe/test_baseline_mse"] = final_metrics["pixel_baseline_test_mse"]
            log_payload["pixel_probe/test_r2"] = final_metrics["pixel_test_r2"]
        if pixel_plot_saved:
            log_payload["plot/pixel_prediction"] = wandb.Image(args.pixel_plot_file)
        wandb.log(log_payload)
        wandb_run.finish()

    train_rollout.close()
    eval_rollout.close()
    probe_train_rollout.close()
    probe_test_rollout.close()
    if pixel_probe_train_rollout is not None:
        pixel_probe_train_rollout.close()
    if pixel_probe_test_rollout is not None:
        pixel_probe_test_rollout.close()
    print("Done.")


if __name__ == "__main__":
    main()
