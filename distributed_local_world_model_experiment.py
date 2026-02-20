import argparse
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def make_generator(seed: int, device: torch.device) -> Optional[torch.Generator]:
    if device.type == "cuda":
        return torch.Generator(device="cuda").manual_seed(seed)
    if device.type == "cpu":
        return torch.Generator().manual_seed(seed)
    # MPS and some other backends may not support explicit generators.
    return None


def build_grid_neighbors(height: int, width: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_agents = height * width
    neighbor_idx = torch.full((num_agents, 4), -1, dtype=torch.long)
    neighbor_mask = torch.zeros((num_agents, 4), dtype=torch.float32)
    edge_delta = torch.zeros((num_agents, 4, 2), dtype=torch.float32)

    denom_y = float(max(1, height - 1))
    denom_x = float(max(1, width - 1))

    for r in range(height):
        for c in range(width):
            i = r * width + c
            candidates = [
                (r - 1, c),
                (r + 1, c),
                (r, c - 1),
                (r, c + 1),
            ]
            for slot, (rr, cc) in enumerate(candidates):
                if 0 <= rr < height and 0 <= cc < width:
                    j = rr * width + cc
                    neighbor_idx[i, slot] = j
                    neighbor_mask[i, slot] = 1.0
                    edge_delta[i, slot, 0] = float(rr - r) / denom_y
                    edge_delta[i, slot, 1] = float(cc - c) / denom_x

    return neighbor_idx.to(device), neighbor_mask.to(device), edge_delta.to(device)


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


def sample_identity_assignments(
    batch_size: int,
    num_agents: int,
    permute_agent_positions: bool,
    device: torch.device,
) -> torch.Tensor:
    if not permute_agent_positions:
        identity_at_pos = np.tile(np.arange(num_agents, dtype=np.int64), (batch_size, 1))
    else:
        identity_at_pos = np.stack(
            [np.random.permutation(num_agents).astype(np.int64) for _ in range(batch_size)],
            axis=0,
        )
    return torch.as_tensor(identity_at_pos, dtype=torch.long, device=device)


def diffusion_step(
    x: torch.Tensor,
    diffusion: float,
    forcing: float,
    noise_std: float,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    neigh_avg = (
        padded[:, :, :-2, 1:-1]
        + padded[:, :, 2:, 1:-1]
        + padded[:, :, 1:-1, :-2]
        + padded[:, :, 1:-1, 2:]
    ) / 4.0

    x_next = x + diffusion * (neigh_avg - x)
    if forcing > 0.0:
        if generator is None:
            forcing_term = torch.rand(x.shape, device=x.device) - 0.5
        else:
            forcing_term = torch.rand(x.shape, generator=generator, device=x.device) - 0.5
        x_next = x_next + forcing * forcing_term
    if noise_std > 0.0:
        if generator is None:
            noise = torch.randn(x.shape, device=x.device) * noise_std
        else:
            noise = torch.randn(x.shape, generator=generator, device=x.device) * noise_std
        x_next = x_next + noise
    return x_next.clamp(0.0, 1.0)


def simulate_diffusion_batch(
    batch_size: int,
    seq_len_plus_one: int,
    height: int,
    width: int,
    diffusion: float,
    forcing: float,
    noise_std: float,
    device: torch.device,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if generator is None:
        x = torch.rand((batch_size, 1, height, width), device=device)
    else:
        x = torch.rand((batch_size, 1, height, width), generator=generator, device=device)
    states = [x[:, 0]]
    for _ in range(seq_len_plus_one - 1):
        x = diffusion_step(x, diffusion, forcing, noise_std, generator)
        states.append(x[:, 0])
    return torch.stack(states, dim=1)


def extract_patches(states: torch.Tensor, patch_radius: int) -> torch.Tensor:
    kernel_size = 2 * patch_radius + 1
    unfolded = F.unfold(states.unsqueeze(1), kernel_size=kernel_size, padding=patch_radius)
    return unfolded.transpose(1, 2).contiguous()


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, hidden_layers: int):
        super().__init__()
        layers = []
        last_dim = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DistributedWorldModel(nn.Module):
    def __init__(
        self,
        num_identities: int,
        patch_dim: int,
        latent_dim: int,
        id_dim: int,
        hidden_dim: int,
        hidden_layers: int,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.id_dim = id_dim

        self.identity_embed = nn.Embedding(num_identities, id_dim)
        self.id_to_latent = nn.Linear(id_dim, latent_dim)

        self.obs_encoder = MLP(patch_dim, latent_dim, hidden_dim, hidden_layers)
        self.msg_encoder = MLP(latent_dim + 2 * id_dim + 2, latent_dim, hidden_dim, max(1, hidden_layers - 1))

        update_in_dim = latent_dim + latent_dim + id_dim + 1
        self.update_cell = nn.GRUCell(update_in_dim, latent_dim)

        self.self_head = MLP(latent_dim, patch_dim, hidden_dim, hidden_layers)
        self.neighbor_head = MLP(latent_dim, 4 * latent_dim, hidden_dim, hidden_layers)

    def identity_features(self, identity_at_pos: torch.Tensor) -> torch.Tensor:
        return self.identity_embed(identity_at_pos)

    def initial_latent(self, id_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_agents, _ = id_features.shape
        z0 = self.id_to_latent(id_features.reshape(batch_size * num_agents, self.id_dim))
        return z0.reshape(batch_size, num_agents, self.latent_dim)

    def aggregate_messages(
        self,
        z_prev: torch.Tensor,
        id_features: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
        edge_delta: torch.Tensor,
        disable_messages: bool,
    ) -> torch.Tensor:
        if disable_messages:
            return torch.zeros_like(z_prev)

        batch_size, num_agents, _ = z_prev.shape
        idx_safe = neighbor_idx.clamp(min=0)

        receiver_ids = id_features
        agg = torch.zeros_like(z_prev)
        for slot in range(idx_safe.shape[1]):
            sender_z = z_prev[:, idx_safe[:, slot], :]
            sender_ids = id_features[:, idx_safe[:, slot], :]
            rel = edge_delta[:, slot, :].view(1, num_agents, 2).expand(batch_size, num_agents, 2)
            msg_in = torch.cat([sender_z, sender_ids, receiver_ids, rel], dim=-1)
            msg = self.msg_encoder(msg_in.reshape(batch_size * num_agents, -1))
            msg = msg.reshape(batch_size, num_agents, self.latent_dim)
            slot_mask = neighbor_mask[:, slot].view(1, num_agents, 1)
            agg = agg + msg * slot_mask

        denom = neighbor_mask.sum(dim=1).clamp(min=1.0).view(1, num_agents, 1)
        return agg / denom

    def step(
        self,
        z_prev: torch.Tensor,
        obs_patches: torch.Tensor,
        obs_mask: torch.Tensor,
        id_features: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
        edge_delta: torch.Tensor,
        disable_messages: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_agents, patch_dim = obs_patches.shape

        obs_embed = self.obs_encoder(obs_patches.reshape(batch_size * num_agents, patch_dim))
        obs_embed = obs_embed.reshape(batch_size, num_agents, self.latent_dim)
        obs_embed = obs_embed * obs_mask.unsqueeze(-1)

        msg = self.aggregate_messages(
            z_prev=z_prev,
            id_features=id_features,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            edge_delta=edge_delta,
            disable_messages=disable_messages,
        )

        update_in = torch.cat([obs_embed, msg, id_features, obs_mask.unsqueeze(-1)], dim=-1)
        z_next = self.update_cell(
            update_in.reshape(batch_size * num_agents, -1),
            z_prev.reshape(batch_size * num_agents, self.latent_dim),
        )
        z_next = z_next.reshape(batch_size, num_agents, self.latent_dim)

        self_pred = self.self_head(z_next.reshape(batch_size * num_agents, self.latent_dim))
        self_pred = self_pred.reshape(batch_size, num_agents, patch_dim)

        neighbor_pred = self.neighbor_head(z_next.reshape(batch_size * num_agents, self.latent_dim))
        neighbor_pred = neighbor_pred.reshape(batch_size, num_agents, 4, self.latent_dim)
        return z_next, self_pred, neighbor_pred


def gather_neighbor_latents(z: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    idx_safe = neighbor_idx.clamp(min=0)
    gathered = []
    for slot in range(idx_safe.shape[1]):
        gathered.append(z[:, idx_safe[:, slot], :].unsqueeze(2))
    return torch.cat(gathered, dim=2)


def compute_sequence_loss(
    model: DistributedWorldModel,
    states: torch.Tensor,
    patch_radius: int,
    base_observer_mask: torch.Tensor,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    edge_delta: torch.Tensor,
    lambda_self: float,
    lambda_neighbor: float,
    blind_reg_weight: float,
    disable_messages: bool,
    permute_agent_positions: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len_plus_one, height, width = states.shape
    num_agents = height * width
    steps = seq_len_plus_one - 1

    identity_at_pos = sample_identity_assignments(
        batch_size=batch_size,
        num_agents=num_agents,
        permute_agent_positions=permute_agent_positions,
        device=states.device,
    )
    obs_mask = base_observer_mask[identity_at_pos]
    id_features = model.identity_features(identity_at_pos)

    z = model.initial_latent(id_features)

    total_self = torch.zeros((), device=states.device)
    total_neighbor = torch.zeros((), device=states.device)
    total_blind_reg = torch.zeros((), device=states.device)

    neighbor_denom = neighbor_mask.sum().clamp(min=1.0) * batch_size

    for t in range(steps):
        obs_t = extract_patches(states[:, t], patch_radius)
        target_patch_next = extract_patches(states[:, t + 1], patch_radius)

        z_next, self_pred, neighbor_pred = model.step(
            z_prev=z,
            obs_patches=obs_t,
            obs_mask=obs_mask,
            id_features=id_features,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            edge_delta=edge_delta,
            disable_messages=disable_messages,
        )

        self_mse = ((self_pred - target_patch_next) ** 2).mean(dim=-1)
        self_denom = obs_mask.sum().clamp(min=1.0)
        loss_self = (self_mse * obs_mask).sum() / self_denom

        target_neighbors = gather_neighbor_latents(z_next.detach(), neighbor_idx)
        neighbor_mse = ((neighbor_pred - target_neighbors) ** 2).mean(dim=-1)
        loss_neighbor = (neighbor_mse * neighbor_mask.view(1, num_agents, 4)).sum() / neighbor_denom

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
    loss = lambda_self * mean_self + lambda_neighbor * mean_neighbor + blind_reg_weight * mean_blind_reg
    return loss, mean_self, mean_neighbor, mean_blind_reg


@torch.no_grad()
def evaluate_model(
    model: DistributedWorldModel,
    eval_batches: int,
    batch_size: int,
    seq_len: int,
    height: int,
    width: int,
    patch_radius: int,
    base_observer_mask: torch.Tensor,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    edge_delta: torch.Tensor,
    lambda_self: float,
    lambda_neighbor: float,
    blind_reg_weight: float,
    disable_messages: bool,
    permute_agent_positions: bool,
    diffusion: float,
    forcing: float,
    noise_std: float,
    device: torch.device,
    generator: Optional[torch.Generator],
) -> Dict[str, float]:
    model.eval()
    loss_total = 0.0
    self_total = 0.0
    neighbor_total = 0.0
    blind_reg_total = 0.0

    for _ in range(eval_batches):
        states = simulate_diffusion_batch(
            batch_size=batch_size,
            seq_len_plus_one=seq_len + 1,
            height=height,
            width=width,
            diffusion=diffusion,
            forcing=forcing,
            noise_std=noise_std,
            device=device,
            generator=generator,
        )

        loss, loss_self, loss_neighbor, loss_blind_reg = compute_sequence_loss(
            model=model,
            states=states,
            patch_radius=patch_radius,
            base_observer_mask=base_observer_mask,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            edge_delta=edge_delta,
            lambda_self=lambda_self,
            lambda_neighbor=lambda_neighbor,
            blind_reg_weight=blind_reg_weight,
            disable_messages=disable_messages,
            permute_agent_positions=permute_agent_positions,
        )
        loss_total += float(loss.item())
        self_total += float(loss_self.item())
        neighbor_total += float(loss_neighbor.item())
        blind_reg_total += float(loss_blind_reg.item())

    denom = float(max(1, eval_batches))
    return {
        "loss": loss_total / denom,
        "self_loss": self_total / denom,
        "neighbor_loss": neighbor_total / denom,
        "blind_reg_loss": blind_reg_total / denom,
    }


@torch.no_grad()
def collect_probe_dataset(
    model: DistributedWorldModel,
    num_batches: int,
    batch_size: int,
    seq_len: int,
    height: int,
    width: int,
    patch_radius: int,
    base_observer_mask: torch.Tensor,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    edge_delta: torch.Tensor,
    disable_messages: bool,
    permute_agent_positions: bool,
    diffusion: float,
    forcing: float,
    noise_std: float,
    device: torch.device,
    generator: Optional[torch.Generator],
    include_agent_latents: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    model.eval()
    global_features = []
    agent_features = []
    targets = []
    obs_masks = []

    num_agents = height * width

    for _ in range(num_batches):
        states = simulate_diffusion_batch(
            batch_size=batch_size,
            seq_len_plus_one=seq_len + 1,
            height=height,
            width=width,
            diffusion=diffusion,
            forcing=forcing,
            noise_std=noise_std,
            device=device,
            generator=generator,
        )

        identity_at_pos = sample_identity_assignments(
            batch_size=batch_size,
            num_agents=num_agents,
            permute_agent_positions=permute_agent_positions,
            device=device,
        )
        obs_mask = base_observer_mask[identity_at_pos]
        id_features = model.identity_features(identity_at_pos)
        z = model.initial_latent(id_features)

        for t in range(seq_len):
            obs_t = extract_patches(states[:, t], patch_radius)
            z, _, _ = model.step(
                z_prev=z,
                obs_patches=obs_t,
                obs_mask=obs_mask,
                id_features=id_features,
                neighbor_idx=neighbor_idx,
                neighbor_mask=neighbor_mask,
                edge_delta=edge_delta,
                disable_messages=disable_messages,
            )
            global_features.append(z.reshape(batch_size, -1).cpu())
            if include_agent_latents:
                agent_features.append(z.cpu())
            targets.append(states[:, t + 1].reshape(batch_size, -1).cpu())
            obs_masks.append(obs_mask.cpu())

    x_global = torch.cat(global_features, dim=0)
    y = torch.cat(targets, dim=0)
    obs_mask_all = torch.cat(obs_masks, dim=0)
    x_agent = torch.cat(agent_features, dim=0) if include_agent_latents else None
    return x_global, x_agent, y, obs_mask_all


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


def reconstruction_split_mse(pred: torch.Tensor, target: torch.Tensor, obs_mask: torch.Tensor) -> Dict[str, float]:
    sq = (pred - target) ** 2
    obs_denom = obs_mask.sum().clamp(min=1.0)
    blind_mask = 1.0 - obs_mask
    blind_denom = blind_mask.sum().clamp(min=1.0)

    obs_mse = float((sq * obs_mask).sum().item() / obs_denom.item())
    blind_mse = float((sq * blind_mask).sum().item() / blind_denom.item())
    return {
        "observed_cells_mse": obs_mse,
        "blind_cells_mse": blind_mse,
    }


def evaluate_agent_probes(
    x_agent_train: torch.Tensor,
    x_agent_test: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    observer_mask: torch.Tensor,
    l2: float,
    max_agents: int,
) -> Dict[str, float]:
    num_agents = x_agent_train.shape[1]
    if max_agents <= 0 or max_agents >= num_agents:
        selected = np.arange(num_agents)
    else:
        selected = np.linspace(0, num_agents - 1, num=max_agents, dtype=int)

    observer_np = observer_mask.cpu().numpy() > 0.5
    observer_mses = []
    blind_mses = []

    for agent_idx in selected:
        w = fit_ridge_regression(x_agent_train[:, agent_idx, :], y_train, l2)
        pred = ridge_predict(x_agent_test[:, agent_idx, :], w)
        mse = mse_value(pred, y_test)
        if observer_np[agent_idx]:
            observer_mses.append(mse)
        else:
            blind_mses.append(mse)

    return {
        "observer_agent_probe_mse": float(np.mean(observer_mses)) if observer_mses else float("nan"),
        "blind_agent_probe_mse": float(np.mean(blind_mses)) if blind_mses else float("nan"),
        "num_selected_agents": int(len(selected)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distributed local prediction with blind-modality masks + identity-conditioned messaging + global probe."
    )
    parser.add_argument("--grid-size", type=int, default=8, help="Grid height/width; agents = grid-size^2.")
    parser.add_argument("--patch-radius", type=int, default=1, help="Radius for local observation patch.")
    parser.add_argument("--observer-frac", type=float, default=0.5, help="Fraction of observing identities.")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--id-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps-per-epoch", type=int, default=40)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-self", type=float, default=1.0)
    parser.add_argument("--lambda-neighbor", type=float, default=1.0)
    parser.add_argument("--blind-reg-weight", type=float, default=1e-3)
    parser.add_argument("--diffusion", type=float, default=0.20)
    parser.add_argument("--forcing", type=float, default=0.05)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--disable-messages", action="store_true", help="Ablation: remove communication.")
    parser.add_argument(
        "--permute-agent-positions",
        action="store_true",
        help="Anti-cheat: each rollout samples a new identity-to-position assignment.",
    )
    parser.add_argument("--probe-train-batches", type=int, default=24)
    parser.add_argument("--probe-test-batches", type=int, default=12)
    parser.add_argument("--probe-l2", type=float, default=1e-2)
    parser.add_argument("--agent-probe-max-agents", type=int, default=0, help="0 = evaluate all agents.")
    parser.add_argument("--skip-agent-probes", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    if args.grid_size < 2:
        raise ValueError("--grid-size must be >= 2")
    if args.patch_radius < 0:
        raise ValueError("--patch-radius must be >= 0")
    if args.seq_len < 1:
        raise ValueError("--seq-len must be >= 1")

    set_seed(args.seed)
    device = resolve_device(args.device)

    height = args.grid_size
    width = args.grid_size
    num_agents = height * width
    patch_dim = (2 * args.patch_radius + 1) ** 2

    neighbor_idx, neighbor_mask, edge_delta = build_grid_neighbors(height, width, device)
    base_observer_mask = sample_observer_mask(num_agents, args.observer_frac, args.seed, device)

    model = DistributedWorldModel(
        num_identities=num_agents,
        patch_dim=patch_dim,
        latent_dim=args.latent_dim,
        id_dim=args.id_dim,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_gen = make_generator(args.seed + 11, device)
    eval_gen = make_generator(args.seed + 17, device)
    probe_train_gen = make_generator(args.seed + 23, device)
    probe_test_gen = make_generator(args.seed + 29, device)

    num_observers = int((base_observer_mask > 0.5).sum().item())
    num_blind = num_agents - num_observers

    print("Experiment: distributed_local_world_model_experiment.py")
    print(f"Device: {device}")
    print(f"Grid size: {height}x{width} (agents={num_agents})")
    print(f"Patch radius: {args.patch_radius} (patch dim={patch_dim})")
    print(f"Observer identities: {num_observers} | Blind identities: {num_blind}")
    print(f"Identity dim: {args.id_dim}")
    print(f"Permute identity-to-position each rollout: {args.permute_agent_positions}")
    print(f"Disable messages: {args.disable_messages}")
    print(
        "Loss weights: "
        f"lambda_self={args.lambda_self}, "
        f"lambda_neighbor={args.lambda_neighbor}, "
        f"blind_reg_weight={args.blind_reg_weight}"
    )
    print(f"Dynamics: diffusion={args.diffusion}, forcing={args.forcing}, noise_std={args.noise_std}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_self = 0.0
        epoch_neighbor = 0.0
        epoch_blind_reg = 0.0

        for _ in range(args.steps_per_epoch):
            states = simulate_diffusion_batch(
                batch_size=args.batch_size,
                seq_len_plus_one=args.seq_len + 1,
                height=height,
                width=width,
                diffusion=args.diffusion,
                forcing=args.forcing,
                noise_std=args.noise_std,
                device=device,
                generator=train_gen,
            )

            loss, loss_self, loss_neighbor, loss_blind_reg = compute_sequence_loss(
                model=model,
                states=states,
                patch_radius=args.patch_radius,
                base_observer_mask=base_observer_mask,
                neighbor_idx=neighbor_idx,
                neighbor_mask=neighbor_mask,
                edge_delta=edge_delta,
                lambda_self=args.lambda_self,
                lambda_neighbor=args.lambda_neighbor,
                blind_reg_weight=args.blind_reg_weight,
                disable_messages=args.disable_messages,
                permute_agent_positions=args.permute_agent_positions,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_self += float(loss_self.item())
            epoch_neighbor += float(loss_neighbor.item())
            epoch_blind_reg += float(loss_blind_reg.item())

        if epoch % args.log_interval == 0 or epoch == 1 or epoch == args.epochs:
            denom = float(args.steps_per_epoch)
            metrics = evaluate_model(
                model=model,
                eval_batches=args.eval_batches,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                height=height,
                width=width,
                patch_radius=args.patch_radius,
                base_observer_mask=base_observer_mask,
                neighbor_idx=neighbor_idx,
                neighbor_mask=neighbor_mask,
                edge_delta=edge_delta,
                lambda_self=args.lambda_self,
                lambda_neighbor=args.lambda_neighbor,
                blind_reg_weight=args.blind_reg_weight,
                disable_messages=args.disable_messages,
                permute_agent_positions=args.permute_agent_positions,
                diffusion=args.diffusion,
                forcing=args.forcing,
                noise_std=args.noise_std,
                device=device,
                generator=eval_gen,
            )
            print(
                f"Epoch {epoch:4d} | "
                f"train loss={epoch_loss / denom:.6f} "
                f"(self={epoch_self / denom:.6f}, nbr={epoch_neighbor / denom:.6f}, blind_reg={epoch_blind_reg / denom:.6f}) | "
                f"eval loss={metrics['loss']:.6f} "
                f"(self={metrics['self_loss']:.6f}, nbr={metrics['neighbor_loss']:.6f}, blind_reg={metrics['blind_reg_loss']:.6f})"
            )

    print("Collecting latent datasets for global reconstruction probe...")
    x_global_train, x_agent_train, y_train, obs_mask_train = collect_probe_dataset(
        model=model,
        num_batches=args.probe_train_batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        height=height,
        width=width,
        patch_radius=args.patch_radius,
        base_observer_mask=base_observer_mask,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        edge_delta=edge_delta,
        disable_messages=args.disable_messages,
        permute_agent_positions=args.permute_agent_positions,
        diffusion=args.diffusion,
        forcing=args.forcing,
        noise_std=args.noise_std,
        device=device,
        generator=probe_train_gen,
        include_agent_latents=not args.skip_agent_probes,
    )
    x_global_test, x_agent_test, y_test, obs_mask_test = collect_probe_dataset(
        model=model,
        num_batches=args.probe_test_batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        height=height,
        width=width,
        patch_radius=args.patch_radius,
        base_observer_mask=base_observer_mask,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        edge_delta=edge_delta,
        disable_messages=args.disable_messages,
        permute_agent_positions=args.permute_agent_positions,
        diffusion=args.diffusion,
        forcing=args.forcing,
        noise_std=args.noise_std,
        device=device,
        generator=probe_test_gen,
        include_agent_latents=not args.skip_agent_probes,
    )

    print("Fitting global linear probe...")
    w_global = fit_ridge_regression(x_global_train, y_train, args.probe_l2)
    train_pred = ridge_predict(x_global_train, w_global)
    test_pred = ridge_predict(x_global_test, w_global)
    global_train_mse = mse_value(train_pred, y_train)
    global_test_mse = mse_value(test_pred, y_test)

    mean_baseline = y_train.mean(dim=0, keepdim=True)
    baseline_test_mse = mse_value(mean_baseline.expand_as(y_test), y_test)
    split_metrics = reconstruction_split_mse(test_pred, y_test, obs_mask_test)

    sse = torch.sum((test_pred - y_test) ** 2)
    sst = torch.sum((y_test - y_test.mean(dim=0, keepdim=True)) ** 2).clamp(min=1e-8)
    global_r2 = float((1.0 - sse / sst).item())

    print("Global probe results (freeze world model; no global training loss used):")
    print(f"- train_mse: {global_train_mse:.6f}")
    print(f"- test_mse: {global_test_mse:.6f}")
    print(f"- test_mse_baseline(mean field): {baseline_test_mse:.6f}")
    print(f"- test_r2: {global_r2:.6f}")
    print(f"- observed_cells_mse: {split_metrics['observed_cells_mse']:.6f}")
    print(f"- blind_cells_mse: {split_metrics['blind_cells_mse']:.6f}")

    if not args.skip_agent_probes and x_agent_train is not None and x_agent_test is not None:
        if args.permute_agent_positions:
            print("Per-agent probes skipped: identity-to-position permutation makes static per-position grouping ill-defined.")
        else:
            print("Fitting per-agent probes (single latent -> full global state)...")
            agent_stats = evaluate_agent_probes(
                x_agent_train=x_agent_train,
                x_agent_test=x_agent_test,
                y_train=y_train,
                y_test=y_test,
                observer_mask=base_observer_mask,
                l2=args.probe_l2,
                max_agents=args.agent_probe_max_agents,
            )
            print("Per-agent probe results:")
            print(f"- selected_agents: {agent_stats['num_selected_agents']}")
            print(f"- observer_agent_probe_mse: {agent_stats['observer_agent_probe_mse']:.6f}")
            print(f"- blind_agent_probe_mse: {agent_stats['blind_agent_probe_mse']:.6f}")
    else:
        print("Per-agent probes skipped.")

    print("Done.")


if __name__ == "__main__":
    main()
