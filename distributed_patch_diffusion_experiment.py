import argparse
import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import wandb
except Exception:
    wandb = None


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
    return None


def torch_rand(shape: Tuple[int, ...], device: torch.device, generator: Optional[torch.Generator]) -> torch.Tensor:
    if generator is None:
        return torch.rand(shape, device=device)
    return torch.rand(shape, device=device, generator=generator)


def torch_randn(shape: Tuple[int, ...], device: torch.device, generator: Optional[torch.Generator]) -> torch.Tensor:
    if generator is None:
        return torch.randn(shape, device=device)
    return torch.randn(shape, device=device, generator=generator)


def torch_randint(
    low: int,
    high: int,
    shape: Tuple[int, ...],
    device: torch.device,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if generator is None:
        return torch.randint(low, high, shape, device=device)
    return torch.randint(low, high, shape, device=device, generator=generator)


def build_grid_neighbors(rows: int, cols: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    num_agents = rows * cols
    neighbor_idx = torch.full((num_agents, 4), -1, dtype=torch.long)
    neighbor_mask = torch.zeros((num_agents, 4), dtype=torch.float32)

    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            for slot, (rr, cc) in enumerate(candidates):
                if 0 <= rr < rows and 0 <= cc < cols:
                    j = rr * cols + cc
                    neighbor_idx[i, slot] = j
                    neighbor_mask[i, slot] = 1.0
    return neighbor_idx.to(device), neighbor_mask.to(device)


def extract_agent_patches(x: torch.Tensor, agent_rows: int, agent_cols: int) -> torch.Tensor:
    batch_size, channels, height, width = x.shape
    patch_h = height // agent_rows
    patch_w = width // agent_cols
    patches = x.view(batch_size, channels, agent_rows, patch_h, agent_cols, patch_w)
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    return patches.view(batch_size, agent_rows * agent_cols, channels * patch_h * patch_w)


def stitch_agent_patches(
    patches: torch.Tensor,
    agent_rows: int,
    agent_cols: int,
    channels: int,
    patch_h: int,
    patch_w: int,
) -> torch.Tensor:
    batch_size = patches.shape[0]
    x = patches.view(batch_size, agent_rows, agent_cols, channels, patch_h, patch_w)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.view(batch_size, channels, agent_rows * patch_h, agent_cols * patch_w)


def sample_smooth_grids(
    batch_size: int,
    height: int,
    width: int,
    smoothing_steps: int,
    device: torch.device,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    x = 2.0 * torch_rand((batch_size, 1, height, width), device, generator) - 1.0
    for _ in range(smoothing_steps):
        padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
        neigh_avg = (
            padded[:, :, :-2, 1:-1]
            + padded[:, :, 2:, 1:-1]
            + padded[:, :, 1:-1, :-2]
            + padded[:, :, 1:-1, 2:]
        ) / 4.0
        x = 0.65 * x + 0.35 * neigh_avg
    return x.clamp(-1.0, 1.0)


def make_alpha_bars(num_steps: int, beta_start: float, beta_end: float, device: torch.device) -> torch.Tensor:
    betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return torch.cat([torch.ones(1, device=device), alpha_bars], dim=0)


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, alpha_bars: torch.Tensor) -> torch.Tensor:
    a_bar = alpha_bars[t].view(-1, 1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise


def timestep_features(t: torch.Tensor, num_steps: int) -> torch.Tensor:
    tau = t.float() / float(max(1, num_steps))
    return torch.stack(
        [
            tau,
            torch.sin(np.pi * tau),
            torch.cos(np.pi * tau),
            torch.sin(2.0 * np.pi * tau),
            torch.cos(2.0 * np.pi * tau),
        ],
        dim=-1,
    )


class LoRALinearPerAgent(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_agents: int, rank: int, alpha: float):
        super().__init__()
        self.rank = rank
        self.scale = float(alpha) / float(max(1, rank))
        self.base = nn.Linear(in_dim, out_dim)

        if rank > 0:
            self.lora_a = nn.Parameter(torch.zeros(num_agents, rank, in_dim))
            self.lora_b = nn.Parameter(torch.zeros(num_agents, out_dim, rank))
            nn.init.normal_(self.lora_a, std=0.02)
            nn.init.zeros_(self.lora_b)
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    def forward(self, x: torch.Tensor, agent_idx: int) -> torch.Tensor:
        y = self.base(x)
        if self.rank <= 0:
            return y
        a = self.lora_a[agent_idx]
        b = self.lora_b[agent_idx]
        delta = (x @ a.t()) @ b.t()
        return y + self.scale * delta


class SharedLoRADiffusionModel(nn.Module):
    def __init__(
        self,
        num_agents: int,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        lora_rank: int,
        lora_alpha: float,
    ):
        super().__init__()
        self.num_agents = num_agents
        dims = [input_dim] + [hidden_dim] * hidden_layers + [output_dim]
        self.layers = nn.ModuleList(
            [
                LoRALinearPerAgent(
                    in_dim=dims[i],
                    out_dim=dims[i + 1],
                    num_agents=num_agents,
                    rank=lora_rank,
                    alpha=lora_alpha,
                )
                for i in range(len(dims) - 1)
            ]
        )

    def forward_agent(self, x: torch.Tensor, agent_idx: int) -> torch.Tensor:
        h = x
        for layer_idx, layer in enumerate(self.layers):
            h = layer(h, agent_idx)
            if layer_idx < len(self.layers) - 1:
                h = F.relu(h)
        return h

    def forward(
        self,
        noisy_patches: torch.Tensor,
        timesteps: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
        num_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_agents, patch_dim = noisy_patches.shape
        if num_agents != self.num_agents:
            raise ValueError("num_agents mismatch in model.forward.")

        idx_safe = neighbor_idx.clamp(min=0)
        neighbor_patches = noisy_patches[:, idx_safe, :]
        neighbor_patches = neighbor_patches * neighbor_mask.view(1, num_agents, 4, 1)
        t_feat = timestep_features(timesteps, num_steps).to(noisy_patches.dtype)

        own_preds = []
        neighbor_preds = []
        for agent_idx in range(num_agents):
            self_patch = noisy_patches[:, agent_idx, :]
            neigh_flat = neighbor_patches[:, agent_idx, :, :].reshape(batch_size, -1)
            neigh_presence = neighbor_mask[agent_idx].view(1, 4).expand(batch_size, 4)
            model_in = torch.cat([self_patch, neigh_flat, neigh_presence, t_feat], dim=-1)

            out = self.forward_agent(model_in, agent_idx)
            own = out[:, :patch_dim]
            neigh = out[:, patch_dim:].reshape(batch_size, 4, patch_dim)
            own_preds.append(own)
            neighbor_preds.append(neigh)

        return torch.stack(own_preds, dim=1), torch.stack(neighbor_preds, dim=1)


def compute_step_loss(
    model: SharedLoRADiffusionModel,
    x_k: torch.Tensor,
    x_km1: torch.Tensor,
    timesteps: torch.Tensor,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    agent_rows: int,
    agent_cols: int,
    num_steps: int,
    lambda_neighbor: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_patches = extract_agent_patches(x_km1, agent_rows, agent_cols)
    noisy_patches = extract_agent_patches(x_k, agent_rows, agent_cols)
    own_pred, neighbor_pred = model(
        noisy_patches=noisy_patches,
        timesteps=timesteps,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        num_steps=num_steps,
    )

    own_loss = F.mse_loss(own_pred, target_patches)

    idx_safe = neighbor_idx.clamp(min=0)
    target_neighbors = target_patches[:, idx_safe, :]
    neighbor_sq = ((neighbor_pred - target_neighbors) ** 2).mean(dim=-1)

    mask = neighbor_mask.view(1, neighbor_mask.shape[0], 4)
    neighbor_denom = mask.sum().clamp(min=1.0) * float(x_k.shape[0])
    neighbor_loss = (neighbor_sq * mask).sum() / neighbor_denom

    loss = own_loss + lambda_neighbor * neighbor_loss
    return loss, own_loss, neighbor_loss


def tensor_to_image(x: torch.Tensor) -> np.ndarray:
    # x is [1, H, W] in [-1, 1]
    img = x.detach().cpu().clamp(-1.0, 1.0)
    img = (img + 1.0) * 0.5
    return img[0].numpy()


def make_eval_visual_panel(x0: torch.Tensor, x_t: torch.Tensor, x_rec: torch.Tensor, t: int) -> Optional[np.ndarray]:
    if plt is None:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    titles = [f"x0 (clean)", f"x_t (t={t})", "x_recon", "abs error"]
    imgs = [
        tensor_to_image(x0),
        tensor_to_image(x_t),
        tensor_to_image(x_rec),
        np.abs(tensor_to_image(x_rec) - tensor_to_image(x0)),
    ]
    vmins = [0.0, 0.0, 0.0, 0.0]
    vmaxs = [1.0, 1.0, 1.0, 1.0]
    for ax, title, img, vmin, vmax in zip(axes, titles, imgs, vmins, vmaxs):
        ax.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    panel = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
    plt.close(fig)
    return panel


def reconstruct_from_xt(
    model: SharedLoRADiffusionModel,
    x_t: torch.Tensor,
    start_step: int,
    num_steps: int,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    agent_rows: int,
    agent_cols: int,
    patch_h: int,
    patch_w: int,
) -> torch.Tensor:
    x = x_t
    batch_size = x.shape[0]
    for step in range(start_step, 0, -1):
        timesteps = torch.full((batch_size,), step, dtype=torch.long, device=x.device)
        patches = extract_agent_patches(x, agent_rows, agent_cols)
        own_pred, _ = model(
            noisy_patches=patches,
            timesteps=timesteps,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            num_steps=num_steps,
        )
        # Reconstruction rule from the prompt: x^{k-1} = cat_i x^{k-1}_i
        x = stitch_agent_patches(own_pred, agent_rows, agent_cols, 1, patch_h, patch_w)
    return x


@torch.no_grad()
def evaluate_model(
    model: SharedLoRADiffusionModel,
    eval_batches: int,
    batch_size: int,
    image_size: int,
    smoothing_steps: int,
    num_steps: int,
    alpha_bars: torch.Tensor,
    neighbor_idx: torch.Tensor,
    neighbor_mask: torch.Tensor,
    agent_rows: int,
    agent_cols: int,
    lambda_neighbor: float,
    device: torch.device,
    generator: Optional[torch.Generator],
    return_visuals: bool = False,
) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]]]:
    model.eval()
    loss_total = 0.0
    own_total = 0.0
    neighbor_total = 0.0
    recon_total = 0.0
    visuals: Optional[Dict[str, np.ndarray]] = None

    patch_h = image_size // agent_rows
    patch_w = image_size // agent_cols

    for _ in range(eval_batches):
        x0 = sample_smooth_grids(
            batch_size=batch_size,
            height=image_size,
            width=image_size,
            smoothing_steps=smoothing_steps,
            device=device,
            generator=generator,
        )
        eps = torch_randn(x0.shape, device=device, generator=generator)
        t = torch_randint(1, num_steps + 1, (batch_size,), device=device, generator=generator)
        x_k = q_sample(x0, t, eps, alpha_bars)
        x_km1 = q_sample(x0, t - 1, eps, alpha_bars)

        loss, own_loss, neighbor_loss = compute_step_loss(
            model=model,
            x_k=x_k,
            x_km1=x_km1,
            timesteps=t,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            agent_rows=agent_rows,
            agent_cols=agent_cols,
            num_steps=num_steps,
            lambda_neighbor=lambda_neighbor,
        )
        loss_total += float(loss.item())
        own_total += float(own_loss.item())
        neighbor_total += float(neighbor_loss.item())

        x_t = q_sample(
            x0,
            torch.full((batch_size,), num_steps, dtype=torch.long, device=device),
            eps,
            alpha_bars,
        )
        x_rec = reconstruct_from_xt(
            model=model,
            x_t=x_t,
            start_step=num_steps,
            num_steps=num_steps,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            agent_rows=agent_rows,
            agent_cols=agent_cols,
            patch_h=patch_h,
            patch_w=patch_w,
        )
        recon_total += float(F.mse_loss(x_rec, x0).item())
        if return_visuals and visuals is None:
            panel = make_eval_visual_panel(x0[0], x_t[0], x_rec[0], num_steps)
            if panel is not None:
                visuals = {"eval_panel": panel}

    denom = float(max(1, eval_batches))
    metrics = {
        "loss": loss_total / denom,
        "own_loss": own_total / denom,
        "neighbor_loss": neighbor_total / denom,
        "recon_mse_from_xT": recon_total / denom,
    }
    return metrics, visuals


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Shared local diffusion denoisers with agent-identity LoRA adapters. "
            "Each agent consumes x^k_i and x^k_N(i), predicts x^{k-1}_i and x^{k-1}_N(i), "
            "and global reconstruction uses only stitched own predictions."
        )
    )
    parser.add_argument("--image-size", type=int, default=16, help="Square image/grid side length.")
    parser.add_argument("--agent-grid-rows", type=int, default=4, help="Rows in agent partition.")
    parser.add_argument("--agent-grid-cols", type=int, default=4, help="Cols in agent partition.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=6)
    parser.add_argument("--num-diffusion-steps", type=int, default=20)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=4e-2)
    parser.add_argument("--smoothing-steps", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--lambda-neighbor", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="emergent-world-models")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-log-every", type=int, default=1, help="Log scalar metrics to W&B every N eval epochs.")
    parser.add_argument("--wandb-vis-every", type=int, default=5, help="Log eval visualization image to W&B every N eval epochs.")
    args = parser.parse_args()

    if args.image_size < 4:
        raise ValueError("--image-size must be >= 4")
    if args.agent_grid_rows < 1 or args.agent_grid_cols < 1:
        raise ValueError("--agent-grid-rows and --agent-grid-cols must be >= 1")
    if args.image_size % args.agent_grid_rows != 0 or args.image_size % args.agent_grid_cols != 0:
        raise ValueError("--image-size must be divisible by --agent-grid-rows and --agent-grid-cols")
    if args.num_diffusion_steps < 1:
        raise ValueError("--num-diffusion-steps must be >= 1")
    if args.lora_rank < 0:
        raise ValueError("--lora-rank must be >= 0")

    set_seed(args.seed)
    device = resolve_device(args.device)
    train_gen = make_generator(args.seed + 11, device)
    eval_gen = make_generator(args.seed + 17, device)

    num_agents = args.agent_grid_rows * args.agent_grid_cols
    patch_h = args.image_size // args.agent_grid_rows
    patch_w = args.image_size // args.agent_grid_cols
    patch_dim = patch_h * patch_w
    max_neighbors = 4
    time_dim = 5
    model_in_dim = patch_dim * (1 + max_neighbors) + max_neighbors + time_dim
    model_out_dim = patch_dim * (1 + max_neighbors)

    neighbor_idx, neighbor_mask = build_grid_neighbors(args.agent_grid_rows, args.agent_grid_cols, device)
    alpha_bars = make_alpha_bars(args.num_diffusion_steps, args.beta_start, args.beta_end, device)

    model = SharedLoRADiffusionModel(
        num_agents=num_agents,
        input_dim=model_in_dim,
        output_dim=model_out_dim,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb requested but package is not installed.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    print("Experiment: distributed_patch_diffusion_experiment.py")
    print(f"Device: {device}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Agent grid: {args.agent_grid_rows}x{args.agent_grid_cols} (agents={num_agents})")
    print(f"Patch size: {patch_h}x{patch_w} (patch_dim={patch_dim})")
    print(f"Diffusion steps: {args.num_diffusion_steps} (beta_start={args.beta_start}, beta_end={args.beta_end})")
    print(f"Model: shared MLP + per-agent LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
    print(f"Loss weights: own=1.0, neighbor={args.lambda_neighbor}")
    print("Reconstruction rule: x^{k-1} = cat_i x^{k-1}_i")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_own = 0.0
        train_neighbor = 0.0

        for _ in range(args.steps_per_epoch):
            x0 = sample_smooth_grids(
                batch_size=args.batch_size,
                height=args.image_size,
                width=args.image_size,
                smoothing_steps=args.smoothing_steps,
                device=device,
                generator=train_gen,
            )
            eps = torch_randn(x0.shape, device=device, generator=train_gen)
            t = torch_randint(1, args.num_diffusion_steps + 1, (args.batch_size,), device=device, generator=train_gen)
            x_k = q_sample(x0, t, eps, alpha_bars)
            x_km1 = q_sample(x0, t - 1, eps, alpha_bars)

            loss, own_loss, neighbor_loss = compute_step_loss(
                model=model,
                x_k=x_k,
                x_km1=x_km1,
                timesteps=t,
                neighbor_idx=neighbor_idx,
                neighbor_mask=neighbor_mask,
                agent_rows=args.agent_grid_rows,
                agent_cols=args.agent_grid_cols,
                num_steps=args.num_diffusion_steps,
                lambda_neighbor=args.lambda_neighbor,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            train_own += float(own_loss.item())
            train_neighbor += float(neighbor_loss.item())

        if epoch % args.log_interval == 0 or epoch == 1 or epoch == args.epochs:
            denom = float(args.steps_per_epoch)
            should_log_vis = args.wandb and args.wandb_vis_every > 0 and (epoch % args.wandb_vis_every == 0 or epoch == 1 or epoch == args.epochs)
            eval_metrics, eval_visuals = evaluate_model(
                model=model,
                eval_batches=args.eval_batches,
                batch_size=args.batch_size,
                image_size=args.image_size,
                smoothing_steps=args.smoothing_steps,
                num_steps=args.num_diffusion_steps,
                alpha_bars=alpha_bars,
                neighbor_idx=neighbor_idx,
                neighbor_mask=neighbor_mask,
                agent_rows=args.agent_grid_rows,
                agent_cols=args.agent_grid_cols,
                lambda_neighbor=args.lambda_neighbor,
                device=device,
                generator=eval_gen,
                return_visuals=should_log_vis,
            )
            print(
                f"Epoch {epoch:4d} | "
                f"train loss={train_loss / denom:.6f} "
                f"(own={train_own / denom:.6f}, nbr={train_neighbor / denom:.6f}) | "
                f"eval loss={eval_metrics['loss']:.6f} "
                f"(own={eval_metrics['own_loss']:.6f}, nbr={eval_metrics['neighbor_loss']:.6f}) | "
                f"eval recon_mse_from_xT={eval_metrics['recon_mse_from_xT']:.6f}"
            )
            if wandb_run is not None and args.wandb_log_every > 0 and (epoch % args.wandb_log_every == 0):
                payload = {
                    "epoch": epoch,
                    "train/loss": train_loss / denom,
                    "train/own_loss": train_own / denom,
                    "train/neighbor_loss": train_neighbor / denom,
                    "eval/loss": eval_metrics["loss"],
                    "eval/own_loss": eval_metrics["own_loss"],
                    "eval/neighbor_loss": eval_metrics["neighbor_loss"],
                    "eval/recon_mse_from_xT": eval_metrics["recon_mse_from_xT"],
                }
                if eval_visuals is not None and "eval_panel" in eval_visuals:
                    payload["eval/visual_panel"] = wandb.Image(eval_visuals["eval_panel"])
                wandb.log(payload)

    if wandb_run is not None:
        wandb_run.finish()
    print("Done.")


if __name__ == "__main__":
    main()
