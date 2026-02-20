import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import wandb
except Exception:
    wandb = None

from gym_distributed_local_world_model_experiment import (
    DistributedGymWorldModel,
    GymBatchRollout,
    append_tag_to_path,
    build_graph,
    compute_sequence_loss,
    evaluate_model,
    sample_identity_assignments,
    sample_observer_mask,
    save_side_by_side_mp4,
    set_seed,
    resolve_device,
)


def infer_obs_image_shape_from_env(env) -> Optional[Tuple[int, int, int]]:
    space = env.observation_space
    spaces = getattr(space, "spaces", None)
    if isinstance(spaces, dict) and "image" in spaces:
        shape = tuple(int(v) for v in spaces["image"].shape)
        if len(shape) == 3:
            return shape
    shape = getattr(space, "shape", None)
    if shape is not None and len(shape) == 3:
        return tuple(int(v) for v in shape)
    return None

def build_image_patch_state_masks_full_cover(
    num_agents: int,
    img_shape: Tuple[int, int, int],
    observer_mask: torch.Tensor,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Tile the full image into EXACTLY n_observers rectangles (no gaps, no overlaps),
    and assign one rectangle to each observer. Non-observers get all zeros.
    """
    h, w, c = img_shape
    obs_dim = h * w * c
    masks = np.zeros((num_agents, obs_dim), dtype=np.float32)

    observer_idx = np.flatnonzero(observer_mask.detach().cpu().numpy() > 0.5)
    n_obs = int(observer_idx.size)
    if n_obs == 0:
        return torch.as_tensor(masks, dtype=torch.float32, device=device)

    rng = np.random.default_rng(seed + 101)
    perm_obs = observer_idx[rng.permutation(n_obs)]

    # Choose number of rows R, then distribute observers across rows:
    # counts per row differ by at most 1, total = n_obs.
    R = int(np.floor(np.sqrt(n_obs)))
    R = max(1, min(R, h))  # can't have more rows than pixels
    # If R is too small, bump until we can distribute nicely (optional; usually fine).
    # We'll just distribute anyway.

    base = n_obs // R
    rem = n_obs % R
    obs_per_row = [base + (1 if r < rem else 0) for r in range(R)]  # sums to n_obs

    # Split height into R strips
    ys = np.linspace(0, h, R + 1, dtype=int)

    def set_rect(agent: int, y0: int, y1: int, x0: int, x1: int):
        # Fill mask for rectangle (y0:y1, x0:x1), all channels
        for y in range(y0, y1):
            row_base = (y * w) * c
            for x in range(x0, x1):
                idx0 = row_base + x * c
                masks[agent, idx0:idx0 + c] = 1.0

    k = 0
    for r in range(R):
        y0, y1 = int(ys[r]), int(ys[r + 1])
        n_in_row = obs_per_row[r]
        if n_in_row == 0:
            continue

        # Split width into n_in_row segments for this row
        xs = np.linspace(0, w, n_in_row + 1, dtype=int)
        for j in range(n_in_row):
            agent = int(perm_obs[k])
            k += 1
            x0, x1 = int(xs[j]), int(xs[j + 1])
            set_rect(agent, y0, y1, x0, x1)

    state_mask = torch.as_tensor(masks, dtype=torch.float32, device=device)

    # Hard sanity checks (fail fast if coverage ever breaks)
    cover = state_mask.sum(dim=0)  # [obs_dim]
    if not torch.all(cover > 0.5):
        missing = int((cover <= 0.5).sum().item())
        raise RuntimeError(f"State mask does not fully cover image! Missing dims: {missing}/{obs_dim}")
    if torch.any(cover > 1.5):
        overlap = int((cover > 1.5).sum().item())
        raise RuntimeError(f"State mask has overlaps! Overlapping dims: {overlap}/{obs_dim}")

    return state_mask
def build_image_patch_state_masks(
    num_agents: int,
    img_shape: Tuple[int, int, int],
    observer_mask: torch.Tensor,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Assign each observer agent one contiguous (rectangular) patch of the image.
    Patch covers all channels. Non-observers get all zeros.
    """
    h, w, c = img_shape
    obs_dim = h * w * c
    masks = np.zeros((num_agents, obs_dim), dtype=np.float32)

    observer_idx = np.flatnonzero(observer_mask.detach().cpu().numpy() > 0.5)
    n_obs = int(observer_idx.size)
    if n_obs == 0:
        return torch.as_tensor(masks, dtype=torch.float32, device=device)

    # Permute observer assignment for reproducibility / fairness
    rng = np.random.default_rng(seed + 101)
    perm_obs = observer_idx[rng.permutation(n_obs)]

    # Choose a near-square grid of patches
    grid_rows = int(np.ceil(np.sqrt(n_obs)))
    grid_cols = int(np.ceil(n_obs / grid_rows))

    # Evenly split pixels into grid_rows x grid_cols rectangles
    ys = np.linspace(0, h, grid_rows + 1, dtype=int)
    xs = np.linspace(0, w, grid_cols + 1, dtype=int)

    def flat_index(y, x, ch):
        return (y * w + x) * c + ch

    for k, agent in enumerate(perm_obs):
        r = k // grid_cols
        cc = k % grid_cols
        if r >= grid_rows:
            break  # should not happen, but safe

        y0, y1 = int(ys[r]), int(ys[r + 1])
        x0, x1 = int(xs[cc]), int(xs[cc + 1])

        # Fill mask for this rectangle (all channels)
        for y in range(y0, y1):
            base = (y * w + x0) * c
            # vectorize x loop a bit
            for x in range(x0, x1):
                idx0 = (y * w + x) * c
                masks[int(agent), idx0:idx0 + c] = 1.0

    return torch.as_tensor(masks, dtype=torch.float32, device=device)


def ensure_rgb_vectors(vec: torch.Tensor, h: int, w: int, c: int) -> torch.Tensor:
    frames = vec.reshape(vec.shape[0], h, w, c)
    if c == 3:
        pass
    elif c == 1:
        frames = frames.repeat_interleave(3, dim=-1)
    elif c > 3:
        frames = frames[..., :3]
    else:
        # Rare case, tile channels up to 3.
        reps = int(np.ceil(3.0 / c))
        frames = frames.repeat(1, 1, 1, reps)[..., :3]
    return frames.permute(0, 3, 1, 2).reshape(frames.shape[0], -1)


def vec_to_rgb_hwc(vec: torch.Tensor, h: int, w: int, c: int) -> torch.Tensor:
    rgb_vec = ensure_rgb_vectors(vec, h=h, w=w, c=c)
    return rgb_vec.reshape(rgb_vec.shape[0], 3, h, w).permute(0, 2, 3, 1)


def ensure_rgb_frames_hwc(frames: torch.Tensor) -> torch.Tensor:
    x = frames.float()
    if x.numel() == 0:
        return x
    if x.max() > 1.0:
        x = x / 255.0
    c = int(x.shape[-1])
    if c == 3:
        return x
    if c == 1:
        return x.repeat(1, 1, 1, 3)
    if c > 3:
        return x[..., :3]
    reps = int(np.ceil(3.0 / c))
    return x.repeat(1, 1, 1, reps)[..., :3]


def resize_rgb_hwc(frames: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    if int(frames.shape[1]) == out_h and int(frames.shape[2]) == out_w:
        return frames
    chw = frames.permute(0, 3, 1, 2)
    out = F.interpolate(chw, size=(out_h, out_w), mode="nearest")
    return out.permute(0, 2, 3, 1)


def frames_hwc_to_chw_vectors(frames: torch.Tensor) -> torch.Tensor:
    return frames.permute(0, 3, 1, 2).reshape(frames.shape[0], -1)


def rescale_for_display(true_vec: torch.Tensor, pred_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # MiniGrid observations are often small normalized symbolic values; auto-rescale for visibility.
    flat = torch.cat([true_vec.reshape(-1), pred_vec.reshape(-1)], dim=0)
    if flat.numel() == 0:
        return true_vec, pred_vec
    q99 = float(torch.quantile(flat, 0.99).item())
    if q99 <= 0.0:
        return true_vec, pred_vec
    scale = 1.0 / q99 if q99 < 0.95 else 1.0
    return torch.clamp(true_vec * scale, 0.0, 1.0), torch.clamp(pred_vec * scale, 0.0, 1.0)


@torch.no_grad()
def collect_direct_observer_pixel_sequences(
    model: DistributedGymWorldModel,
    rollout: GymBatchRollout,
    pixel_horizon: int,
    device: torch.device,
    base_observer_mask: torch.Tensor,
    base_state_mask: torch.Tensor,
    graph,
    disable_messages: bool,
    permute_agent_positions: bool,
    video_env_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states_np, actions_np, frames_np = rollout.sample_rollout(seq_len_plus_one=pixel_horizon + 1, include_frames=True)
    if frames_np is None:
        raise RuntimeError(
            "Direct pixel visualization needs render frames, but rollout returned none. "
            "Use an environment with rgb_array rendering."
        )
    states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)
    b, _, obs_dim = states.shape
    num_agents = base_observer_mask.shape[0]

    if video_env_index < 0 or video_env_index >= b:
        raise ValueError(f"--pixel-video-env-index must be in [0, {b - 1}] for current pixel eval batch.")

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

    true_seq = []
    pred_seq = []
    for t in range(pixel_horizon):
        state_t = states[:, t]
        action_t = actions[:, t]

        # Each agent receives only its patch (observers) or nothing (non-observers)
        state_local = state_t.unsqueeze(1).expand(b, num_agents, obs_dim) * state_mask

        # Actions are global: give them to ALL agents (so blind agents can still predict neighbors well)
        action_local = action_t.unsqueeze(1).expand(b, num_agents, action_t.shape[-1])

        z, self_pred, _ = model.step(
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

        # Stitch predicted patches:
        # take each observer's prediction ONLY on its own patch, then sum into full frame.
        numer = (self_pred * state_mask).sum(dim=1)
        denom = state_mask.sum(dim=1).clamp(min=1e-6)
        merged_pred = numer / denom  # per-dim average if overlaps; normally denom is 0 or 1

        pred_seq.append(merged_pred[video_env_index].detach().cpu())
        true_seq.append(states[:, t + 1][video_env_index].detach().cpu())

    true_obs = torch.stack(true_seq, dim=0)
    pred_obs = torch.stack(pred_seq, dim=0)
    true_render = torch.as_tensor(frames_np[video_env_index], dtype=torch.float32)
    return true_obs, pred_obs, true_render


def save_direct_pixel_plot(
    true_frames: torch.Tensor,
    pred_frames: torch.Tensor,
    out_path: str,
) -> None:
    if plt is None:
        return
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    true_img = np.clip(true_frames.numpy(), 0.0, 1.0)
    pred_img = np.clip(pred_frames.numpy(), 0.0, 1.0)
    t = true_img.shape[0]

    fig, axes = plt.subplots(2, t, figsize=(2.2 * t, 4.4))
    if t == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for i in range(t):
        axes[0, i].imshow(true_img[i])
        axes[0, i].set_title(f"True t+{i+1}", fontsize=9)
        axes[0, i].axis("off")
        axes[1, i].imshow(pred_img[i])
        axes[1, i].set_title(f"Pred t+{i+1}", fontsize=9)
        axes[1, i].axis("off")
    fig.suptitle("Direct Observer Pixel Predictions", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_direct_pixel_eval(
    model: DistributedGymWorldModel,
    rollout: GymBatchRollout,
    pixel_horizon: int,
    device: torch.device,
    base_observer_mask: torch.Tensor,
    base_state_mask: torch.Tensor,
    graph,
    disable_messages: bool,
    permute_agent_positions: bool,
    img_shape: Tuple[int, int, int],
    plot_path: str,
    save_mp4: bool,
    mp4_path: str,
    mp4_fps: int,
    video_env_index: int,
    display_auto_rescale: bool,
) -> Dict[str, float]:
    true_obs, pred_obs, true_render = collect_direct_observer_pixel_sequences(
        model=model,
        rollout=rollout,
        pixel_horizon=pixel_horizon,
        device=device,
        base_observer_mask=base_observer_mask,
        base_state_mask=base_state_mask,
        graph=graph,
        disable_messages=disable_messages,
        permute_agent_positions=permute_agent_positions,
        video_env_index=video_env_index,
    )
    h_obs, w_obs, c_obs = img_shape
    if display_auto_rescale:
        _, pred_disp = rescale_for_display(true_obs, pred_obs)
    else:
        pred_disp = pred_obs
    mse = float(torch.mean((pred_obs - true_obs) ** 2).item())
    baseline = true_obs.mean(dim=0, keepdim=True)
    baseline_mse = float(torch.mean((baseline.expand_as(true_obs) - true_obs) ** 2).item())
    sse = torch.sum((pred_obs - true_obs) ** 2)
    sst = torch.sum((true_obs - true_obs.mean(dim=0, keepdim=True)) ** 2).clamp(min=1e-8)
    r2 = float((1.0 - sse / sst).item())

    true_frames_disp = ensure_rgb_frames_hwc(true_render)
    pred_frames_disp = vec_to_rgb_hwc(pred_disp, h=h_obs, w=w_obs, c=c_obs)
    pred_frames_disp = ensure_rgb_frames_hwc(pred_frames_disp)
    pred_frames_disp = resize_rgb_hwc(
        pred_frames_disp, out_h=int(true_frames_disp.shape[1]), out_w=int(true_frames_disp.shape[2])
    )

    save_direct_pixel_plot(
        true_frames=true_frames_disp.cpu(),
        pred_frames=pred_frames_disp.cpu(),
        out_path=plot_path,
    )

    if save_mp4:
        true_rgb = frames_hwc_to_chw_vectors(true_frames_disp.cpu())
        pred_rgb = frames_hwc_to_chw_vectors(pred_frames_disp.cpu())
        save_side_by_side_mp4(
            true_frames_vec=true_rgb,
            pred_frames_vec=pred_rgb,
            pixel_height=int(true_frames_disp.shape[1]),
            pixel_width=int(true_frames_disp.shape[2]),
            output_path=mp4_path,
            fps=mp4_fps,
        )

    return {
        "pixel_direct/test_mse": mse,
        "pixel_direct/test_baseline_mse": baseline_mse,
        "pixel_direct/test_r2": r2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Variant: observers predict local next observation + neighbor next latent; others predict only neighbors."
    )
    parser.add_argument("--env", type=str, default="MiniGrid-Dynamic-Obstacles-16x16-v0")
    parser.add_argument("--agents", type=int, default=32)
    parser.add_argument("--graph", type=str, default="sphere", choices=["ring", "line", "grid", "torus", "sphere"])
    parser.add_argument("--graph-rows", type=int, default=8)
    parser.add_argument("--graph-cols", type=int, default=4)
    parser.add_argument("--observer-frac", type=float, default=0.5)
    parser.add_argument("--observer-placement", type=str, default="cluster2d", choices=["auto", "random", "cluster2d"])
    parser.add_argument(
        "--minigrid-fully-obs",
        dest="minigrid_fully_obs",
        action="store_true",
        help="For MiniGrid envs, use full-grid observation (16x16x3) instead of 7x7 partial view. Default: on.",
    )
    parser.add_argument(
        "--no-minigrid-fully-obs",
        dest="minigrid_fully_obs",
        action="store_false",
        help="Disable MiniGrid full-grid observation wrapping.",
    )
    parser.add_argument("--observer-full-view", action="store_true", help="Observers see full local observation vector.")
    parser.add_argument(
        "--min-obs-dims",
        type=int,
        default=2,
        help="Compatibility flag in this variant (state coverage is split evenly across observers).",
    )
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
    parser.add_argument("--pixel-horizon", type=int, default=8)
    parser.add_argument("--pixel-eval-every", type=int, default=1)
    parser.add_argument("--pixel-plot-file", type=str, default="outputs/direct_observer_pixels.png")
    parser.add_argument("--save-pixel-mp4", action="store_true")
    parser.add_argument("--pixel-mp4-prefix", type=str, default="outputs/direct_observer_pixels")
    parser.add_argument("--pixel-video-fps", type=int, default=3)
    parser.add_argument("--pixel-video-env-index", type=int, default=0)
    parser.add_argument(
        "--display-auto-rescale",
        dest="display_auto_rescale",
        action="store_true",
        help="Display-only contrast rescaling for PNG/MP4 (default: on).",
    )
    parser.add_argument(
        "--no-display-auto-rescale",
        dest="display_auto_rescale",
        action="store_false",
        help="Disable display-only contrast rescaling.",
    )
    parser.set_defaults(display_auto_rescale=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="emergent-world-models")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.set_defaults(minigrid_fully_obs=True)
    args = parser.parse_args()

    if args.pixel_horizon < 1:
        raise ValueError("--pixel-horizon must be >= 1")
    if args.pixel_eval_every < 1:
        raise ValueError("--pixel-eval-every must be >= 1")

    set_seed(args.seed)
    device = resolve_device(args.device)

    train_rollout = GymBatchRollout(
        args.env,
        args.batch_size,
        seed=args.seed + 11,
        frame_skip=args.frame_skip,
        minigrid_fully_obs=args.minigrid_fully_obs,
    )
    eval_rollout = GymBatchRollout(
        args.env,
        args.batch_size,
        seed=args.seed + 17,
        frame_skip=args.frame_skip,
        minigrid_fully_obs=args.minigrid_fully_obs,
    )
    pixel_rollout = GymBatchRollout(
        args.env,
        args.batch_size,
        seed=args.seed + 23,
        frame_skip=args.frame_skip,
        render_mode="rgb_array",
        minigrid_fully_obs=args.minigrid_fully_obs,
    )

    img_shape = infer_obs_image_shape_from_env(train_rollout.envs[0])
    if img_shape is None:
        raise RuntimeError(
            "Direct pixel eval requires an image-like observation space (H,W,C). "
            "This env observation is not image-shaped."
        )
    obs_dim = train_rollout.obs_dim
    if int(np.prod(img_shape)) != obs_dim:
        raise RuntimeError(
            f"Direct pixel eval expects obs_dim == H*W*C ({np.prod(img_shape)}), got {obs_dim}. "
            "Use an env where flattened observation corresponds to image observation."
        )

    graph = build_graph(args.agents, args.graph, device=device, graph_rows=args.graph_rows, graph_cols=args.graph_cols)
    base_observer_mask = sample_observer_mask(
        args.agents,
        args.observer_frac,
        args.seed,
        device,
        graph=graph,
        observer_placement=args.observer_placement,
    )
    if int(base_observer_mask.sum().item()) == 0:
        raise ValueError("This variant requires at least one observer so full image can be partitioned across observers.")
    #base_state_mask = build_image_patch_state_masks(
    #    num_agents=args.agents,
    #    img_shape=img_shape,
    #    observer_mask=base_observer_mask,
    #    seed=args.seed,
    #    device=device,
    #)
    base_state_mask = build_image_patch_state_masks_full_cover(
        num_agents=args.agents,
        img_shape=img_shape,
        observer_mask=base_observer_mask,
        seed=args.seed,
        device=device,
    )
    if args.observer_full_view:
        raise ValueError(
            "--observer-full-view is incompatible with this variant: "
            "observer inputs are enforced as a disjoint full-image patch partition."
        )

    model = DistributedGymWorldModel(
        num_agents=args.agents,
        obs_dim=obs_dim,
        action_dim=train_rollout.action_dim,
        latent_dim=args.latent_dim,
        id_dim=args.id_dim,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        edge_feat_dim=graph.edge_feat.shape[-1],
        max_neighbors=graph.neighbor_idx.shape[1],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb requested but package is not installed.")
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    print("Experiment: gym_distributed_observer_direct_pixel_eval.py")
    print(f"Env: {args.env}")
    print(f"Device: {device}")
    print(f"Obs dim: {obs_dim} | Action dim: {train_rollout.action_dim}")
    print(f"Graph: {args.graph}")
    print(f"Observers: {int(base_observer_mask.sum().item())} | Blind: {args.agents - int(base_observer_mask.sum().item())}")
    observer_rows = base_state_mask[base_observer_mask > 0.5]
    if observer_rows.numel() > 0:
        per_obs = observer_rows.sum(dim=1)
        covered = float((observer_rows.sum(dim=0) > 0.5).float().mean().item())
        print(
            "Observer state split: spatial patches "
            f"(dims/observer min={int(per_obs.min().item())}, max={int(per_obs.max().item())}, "
            f"global_coverage={covered:.3f})"
        )
    print(f"Pixel horizon: {args.pixel_horizon}")
    print(f"Save pixel MP4s: {args.save_pixel_mp4}")

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

        eval_metrics = evaluate_model(
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

        if epoch % args.log_interval == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"Epoch {epoch:4d} | "
                f"train={train_total:.6f} (self={train_self:.6f}, nbr={train_nbr:.6f}, blind={train_blind:.6f}) | "
                f"eval={eval_metrics['loss']:.6f} (self={eval_metrics['self_loss']:.6f}, nbr={eval_metrics['neighbor_loss']:.6f}, blind={eval_metrics['blind_reg_loss']:.6f})"
            )

        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_total,
                    "train/self_loss": train_self,
                    "train/neighbor_loss": train_nbr,
                    "train/blind_reg_loss": train_blind,
                    "eval/loss": eval_metrics["loss"],
                    "eval/self_loss": eval_metrics["self_loss"],
                    "eval/neighbor_loss": eval_metrics["neighbor_loss"],
                    "eval/blind_reg_loss": eval_metrics["blind_reg_loss"],
                }
            )

        if epoch % args.pixel_eval_every == 0:
            tag = f"epoch{epoch:04d}"
            plot_path = append_tag_to_path(args.pixel_plot_file, tag)
            mp4_path = append_tag_to_path(f"{args.pixel_mp4_prefix}.mp4", tag)
            pixel_metrics = run_direct_pixel_eval(
                model=model,
                rollout=pixel_rollout,
                pixel_horizon=args.pixel_horizon,
                device=device,
                base_observer_mask=base_observer_mask,
                base_state_mask=base_state_mask,
                graph=graph,
                disable_messages=args.disable_messages,
                permute_agent_positions=args.permute_agent_positions,
                img_shape=img_shape,
                plot_path=plot_path,
                save_mp4=args.save_pixel_mp4,
                mp4_path=mp4_path,
                mp4_fps=args.pixel_video_fps,
                video_env_index=args.pixel_video_env_index,
                display_auto_rescale=args.display_auto_rescale,
            )
            print(
                f"Direct pixel eval epoch {epoch}: "
                f"mse={pixel_metrics['pixel_direct/test_mse']:.6f} "
                f"baseline={pixel_metrics['pixel_direct/test_baseline_mse']:.6f} "
                f"r2={pixel_metrics['pixel_direct/test_r2']:.6f}"
            )

            if wandb_run is not None:
                payload = {"epoch": epoch}
                payload.update(pixel_metrics)
                if plt is not None and os.path.exists(plot_path):
                    payload["pixel_direct/plot"] = wandb.Image(plot_path)
                if args.save_pixel_mp4 and os.path.exists(mp4_path):
                    payload["pixel_direct/video"] = wandb.Video(mp4_path, format="mp4")
                wandb.log(payload)

    train_rollout.close()
    eval_rollout.close()
    pixel_rollout.close()
    if wandb_run is not None:
        wandb_run.finish()
    print("Done.")


if __name__ == "__main__":
    main()
