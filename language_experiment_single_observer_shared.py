import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SharedAgent(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        state_dim: int,
        max_neighbors: int,
        num_agents: int,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.max_neighbors = max_neighbors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = MLP(embed_dim, state_dim, hidden_dim, hidden_layers)
        self.agent_embedding = nn.Embedding(num_agents, state_dim)
        feat_dim = state_dim * (1 + max_neighbors)
        self.state_updater = MLP(feat_dim, state_dim, hidden_dim, hidden_layers)
        self.token_predictor = MLP(feat_dim, vocab_size, hidden_dim, hidden_layers)
        self.state_predictor = MLP(feat_dim, state_dim * max_neighbors, hidden_dim, hidden_layers)

    def encode_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(token_ids)
        return self.encoder(emb)

    def encode_zero(self, batch_size: int, device: torch.device) -> torch.Tensor:
        zero_embed = torch.zeros((batch_size, self.embedding.embedding_dim), device=device)
        return self.encoder(zero_embed)

    def _features(self, self_states: torch.Tensor, neighbor_states: torch.Tensor, agent_ids: torch.Tensor) -> torch.Tensor:
        # self_states: [A, B, D], neighbor_states: [A, B, N, D], agent_ids: [A]
        agent_bias = self.agent_embedding(agent_ids).unsqueeze(1)  # [A, 1, D]
        self_with_id = self_states + agent_bias
        flat_neighbors = neighbor_states.reshape(self_states.shape[0], self_states.shape[1], -1)
        return torch.cat([self_with_id, flat_neighbors], dim=-1)

    def form_states(self, input_states: torch.Tensor, prev_neighbor_states: torch.Tensor, agent_ids: torch.Tensor) -> torch.Tensor:
        x = self._features(input_states, prev_neighbor_states, agent_ids)
        a, b, _ = x.shape
        out = self.state_updater(x.reshape(a * b, -1))
        return out.reshape(a, b, self.state_dim)

    def predict_next_token_logits(
        self, self_states: torch.Tensor, neighbor_states: torch.Tensor, agent_ids: torch.Tensor
    ) -> torch.Tensor:
        x = self._features(self_states, neighbor_states, agent_ids)
        a, b, _ = x.shape
        out = self.token_predictor(x.reshape(a * b, -1))
        return out.reshape(a, b, -1)

    def predict_neighbor_states(
        self, self_states: torch.Tensor, neighbor_states: torch.Tensor, agent_ids: torch.Tensor
    ) -> torch.Tensor:
        x = self._features(self_states, neighbor_states, agent_ids)
        a, b, _ = x.shape
        out = self.state_predictor(x.reshape(a * b, -1))
        return out.reshape(a, b, self.max_neighbors, self.state_dim)



def build_graph(num_agents: int, graph_type: str, neighbors: int) -> Graph:
    if neighbors < 1 and graph_type not in {"full"}:
        raise ValueError("neighbors must be >= 1 for non-full graphs")
    if neighbors > num_agents - 1 and graph_type not in {"full"}:
        raise ValueError("neighbors must be <= num_agents - 1")

    if graph_type == "ring":
        out = []
        for i in range(num_agents):
            out.append([(i - 1) % num_agents, (i + 1) % num_agents])
        return Graph(out)
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
        out = []
        for i in range(num_agents):
            out.append([j for j in range(num_agents) if j != i])
        return Graph(out)
    if graph_type == "random":
        out = []
        for i in range(num_agents):
            choices = [j for j in range(num_agents) if j != i]
            out.append(random.sample(choices, k=min(neighbors, len(choices))))
        return Graph(out)
    if graph_type == "dense":
        out = []
        for i in range(num_agents):
            out.append([((i + d) % num_agents) for d in range(1, neighbors + 1)])
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



def build_vocab(sentences: List[str]) -> Tuple[dict, list]:
    tokens = []
    for s in sentences:
        tokens.extend(s.strip().lower().split())
    vocab = ["<pad>", "<unk>", "<bos>", "<eos>"] + sorted(set(tokens))
    stoi = {tok: i for i, tok in enumerate(vocab)}
    return stoi, vocab



def encode_sentences(sentences: List[str], stoi: dict) -> List[List[int]]:
    out = []
    for s in sentences:
        words = s.strip().lower().split()
        ids = [stoi["<bos>"]] + [stoi.get(w, stoi["<unk>"]) for w in words] + [stoi["<eos>"]]
        out.append(ids)
    return out



def sample_windows(seqs: List[List[int]], batch_size: int, window_len: int) -> np.ndarray:
    if not seqs:
        raise ValueError("No sequences available for sampling")
    windows = np.zeros((batch_size, window_len), dtype=np.int64)
    for i in range(batch_size):
        seq = random.choice(seqs)
        if len(seq) < window_len:
            raise ValueError("Sequence too short for requested window length")
        max_start = len(seq) - window_len
        start = random.randrange(0, max_start + 1) if max_start > 0 else 0
        window = seq[start : start + window_len]
        windows[i] = np.asarray(window, dtype=np.int64)
    return windows



def load_torchtext_sentences(dataset: str, root: str | None, split: str) -> List[str]:
    try:
        from torchtext.datasets import PennTreebank, WikiText2
    except Exception as exc:
        raise RuntimeError(
            "torchtext is required for real datasets. Install with: conda install -c conda-forge torchtext"
        ) from exc

    if dataset == "wikitext2":
        dataset_cls = WikiText2
    elif dataset == "ptb":
        dataset_cls = PennTreebank
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    try:
        ds = dataset_cls(root=root, split=split)
    except TypeError:
        ds = dataset_cls(root=root)
        if isinstance(ds, (tuple, list)):
            split_map = {"train": 0, "valid": 1, "test": 2}
            ds = ds[split_map.get(split, 0)]

    lines: List[str] = []
    for item in ds:
        line = item[0] if isinstance(item, (tuple, list)) else item
        if not isinstance(line, str):
            line = str(line)
        line = line.strip()
        if line:
            lines.append(line)
    return lines



def build_neighbor_tensors(graph: Graph, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    num_agents = len(graph.neighbors)
    max_neighbors = max(len(n) for n in graph.neighbors)
    idx = torch.zeros((num_agents, max_neighbors), dtype=torch.long, device=device)
    mask = torch.zeros((num_agents, max_neighbors), dtype=torch.float32, device=device)
    for i, neigh in enumerate(graph.neighbors):
        if len(neigh) == 0:
            continue
        idx[i, : len(neigh)] = torch.as_tensor(neigh, dtype=torch.long, device=device)
        mask[i, : len(neigh)] = 1.0
    return idx, mask



def gather_neighbor_states(states: torch.Tensor, neighbor_idx: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
    # states: [A, B, D], neighbor_idx: [A, N], neighbor_mask: [A, N]
    gathered = states[neighbor_idx]  # [A, N, B, D]
    gathered = gathered.permute(0, 2, 1, 3).contiguous()  # [A, B, N, D]
    return gathered * neighbor_mask[:, None, :, None]



def masked_neighbor_mse(
    pred: torch.Tensor, target: torch.Tensor, neighbor_mask: torch.Tensor, batch_size: int, state_dim: int
) -> torch.Tensor:
    # pred/target: [A, B, N, D], neighbor_mask: [A, N]
    diff2 = (pred - target).pow(2) * neighbor_mask[:, None, :, None]
    denom = neighbor_mask.sum() * float(batch_size * state_dim)
    denom = torch.clamp(denom, min=1.0)
    return diff2.sum() / denom



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=16)
    parser.add_argument(
        "--graph",
        type=str,
        default="ring",
        choices=["ring", "ring-k", "line", "full", "random", "dense", "grid", "star"],
    )
    parser.add_argument("--neighbors", type=int, default=None)
    parser.add_argument("--degree", type=int, default=4)  # deprecated alias for neighbors
    parser.add_argument("--observer-agent", type=int, default=0)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--unroll-steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--compile", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--amp", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--corpus-file", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="toy", choices=["toy", "wikitext2", "ptb"])
    parser.add_argument("--sequence-mode", type=str, default="stream", choices=["stream", "sentence"])
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--dataset-split", type=str, default="train")
    args = parser.parse_args()

    if args.agents < 2:
        raise ValueError("--agents must be >= 2 for neighbor-state prediction")
    if args.unroll_steps < 1:
        raise ValueError("--unroll-steps must be >= 1")
    if args.eval_batches < 0:
        raise ValueError("--eval-batches must be >= 0")
    if args.observer_agent < 0 or args.observer_agent >= args.agents:
        raise ValueError("--observer-agent must be in [0, agents-1]")

    set_seed(args.seed)

    if args.corpus_file:
        with open(args.corpus_file, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
    elif args.dataset != "toy":
        sentences = load_torchtext_sentences(args.dataset, args.dataset_root, args.dataset_split)
    else:
        sentences = [
            "the cat sat on the mat",
            "the quick brown fox jumps over the lazy dog",
            "language models learn to predict the next token",
            "emergent behavior can arise from simple local rules",
            "multi agent systems share information through a graph",
            "we test observers and non observers on prediction",
            "the world model summarizes past observations",
            "neural networks can approximate complex dynamics",
            "this is a small synthetic corpus for experiments",
            "each agent sees only part of the state",
        ]

    stoi, vocab = build_vocab(sentences)
    seqs = encode_sentences(sentences, stoi)
    vocab_size = len(vocab)

    if args.sequence_mode == "stream":
        stream = []
        for s in seqs:
            stream.extend(s)
        seqs_for_sampling = [stream]
    else:
        seqs_for_sampling = seqs

    window_len = args.unroll_steps + 1
    valid_seqs = [s for s in seqs_for_sampling if len(s) >= window_len]
    if not valid_seqs:
        raise ValueError(
            "No sequences long enough for unroll_steps. Provide longer sentences, use --sequence-mode stream, or reduce --unroll-steps."
        )

    k_neighbors = args.neighbors if args.neighbors is not None else args.degree
    graph = build_graph(args.agents, args.graph, k_neighbors)
    if any(len(n) == 0 for n in graph.neighbors):
        raise ValueError("Each agent must have at least one neighbor for this experiment")

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
    print("Using device:", device)

    neighbor_idx_t, neighbor_mask_t = build_neighbor_tensors(graph, device)
    max_neighbors = int(neighbor_idx_t.shape[1])

    shared_agent = SharedAgent(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        state_dim=args.state_dim,
        max_neighbors=max_neighbors,
        num_agents=args.agents,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
    ).to(device)

    use_compile = args.compile == "on" or (args.compile == "auto" and hasattr(torch, "compile"))
    if use_compile:
        try:
            shared_agent = torch.compile(shared_agent)
        except Exception as exc:
            print(f"Warning: torch.compile failed ({exc}); continuing without compile")
            use_compile = False

    use_amp = device.type == "cuda" and args.amp != "off"
    if args.amp == "on" and device.type != "cuda":
        print("Warning: --amp on requested but device is not CUDA; disabling AMP")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    opt = torch.optim.Adam(shared_agent.parameters(), lr=args.lr)

    stats_model = shared_agent._orig_mod if hasattr(shared_agent, "_orig_mod") else shared_agent
    total_params = sum(p.numel() for p in stats_model.parameters())
    trainable_params = sum(p.numel() for p in stats_model.parameters() if p.requires_grad)
    token_embedding_params = stats_model.embedding.weight.numel()
    agent_embedding_params = stats_model.agent_embedding.weight.numel()
    encoder_params = sum(p.numel() for p in stats_model.encoder.parameters())
    state_updater_params = sum(p.numel() for p in stats_model.state_updater.parameters())
    token_head_params = sum(p.numel() for p in stats_model.token_predictor.parameters())
    state_head_params = sum(p.numel() for p in stats_model.state_predictor.parameters())
    print("Vocab size:", vocab_size)
    print("Agents:", args.agents)
    print("Observer agent:", args.observer_agent)
    print("Graph:", args.graph)
    if args.graph not in {"full", "ring", "line", "grid", "star"}:
        print("Neighbors per agent:", k_neighbors)
    print("Unroll steps:", args.unroll_steps)
    print("Eval batches (autoregressive):", args.eval_batches)
    print("Model overview:")
    print("  Shared agent weights across all positions")
    print("  Per-agent learned identity embedding added to state features")
    print("  All agents predict neighbor next states; observer additionally predicts next token")
    print(f"  torch.compile active: {use_compile}")
    print(f"  AMP active: {use_amp}")
    print(f"  Trainable parameters: {trainable_params:,} / total: {total_params:,}")
    print("  Parameter breakdown:")
    print(f"    token embedding: {token_embedding_params:,}")
    print(f"    agent identity embedding: {agent_embedding_params:,}")
    print(f"    encoder: {encoder_params:,}")
    print(f"    state updater: {state_updater_params:,}")
    print(f"    token head: {token_head_params:,}")
    print(f"    neighbor-state head: {state_head_params:,}")

    agent_ids_t = torch.arange(args.agents, device=device, dtype=torch.long)
    observer_id_t = torch.as_tensor([args.observer_agent], device=device, dtype=torch.long)

    def autoregressive_token_eval() -> Tuple[float, float]:
        if args.eval_batches == 0:
            return float("nan"), float("nan")

        shared_agent.eval()
        total_ce = 0.0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for _ in range(args.eval_batches):
                windows = sample_windows(valid_seqs, args.batch_size, window_len)
                tokens = torch.as_tensor(windows, dtype=torch.long, device=device)  # [B, T+1]

                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                    zero_state = shared_agent.encode_zero(args.batch_size, device)  # [B, D]
                    base_inputs = zero_state.unsqueeze(0).expand(args.agents, -1, -1)  # [A, B, D]
                    prev_states = torch.zeros((args.agents, args.batch_size, args.state_dim), device=device)

                    input_t0 = base_inputs.clone()
                    input_t0[args.observer_agent] = shared_agent.encode_tokens(tokens[:, 0])
                    prev_neighbors = gather_neighbor_states(prev_states, neighbor_idx_t, neighbor_mask_t)
                    curr_states = shared_agent.form_states(input_t0, prev_neighbors, agent_ids_t)

                    for t in range(args.unroll_steps):
                        curr_neighbors = gather_neighbor_states(curr_states, neighbor_idx_t, neighbor_mask_t)
                        obs_state = curr_states[args.observer_agent : args.observer_agent + 1]
                        obs_neighbors = curr_neighbors[args.observer_agent : args.observer_agent + 1]
                        obs_logits = shared_agent.predict_next_token_logits(
                            obs_state, obs_neighbors, observer_id_t
                        ).squeeze(0)

                        ce = F.cross_entropy(obs_logits, tokens[:, t + 1], reduction="mean")
                        total_ce += float(ce.item()) * args.batch_size
                        pred_tokens = torch.argmax(obs_logits, dim=-1)
                        total_correct += int((pred_tokens == tokens[:, t + 1]).sum().item())
                        total_tokens += args.batch_size

                        next_input = base_inputs.clone()
                        next_input[args.observer_agent] = shared_agent.encode_tokens(pred_tokens)
                        next_states = shared_agent.form_states(next_input, curr_neighbors, agent_ids_t)
                        curr_states = next_states

        shared_agent.train()
        avg_ce = total_ce / max(1, total_tokens)
        avg_acc = total_correct / max(1, total_tokens)
        return avg_ce, avg_acc

    for ep in range(1, args.epochs + 1):
        total_metric = 0.0
        total_count = 0
        obs_ce = 0.0
        obs_count = 0
        neighbor_mse = 0.0
        neighbor_count = 0

        for _ in range(args.steps_per_epoch):
            windows = sample_windows(valid_seqs, args.batch_size, window_len)
            tokens = torch.as_tensor(windows, dtype=torch.long, device=device)  # [B, T+1]

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                zero_state = shared_agent.encode_zero(args.batch_size, device)  # [B, D]
                base_inputs = zero_state.unsqueeze(0).expand(args.agents, -1, -1)  # [A, B, D]

                obs_encoded = [shared_agent.encode_tokens(tokens[:, t]) for t in range(window_len)]
                prev_states = torch.zeros((args.agents, args.batch_size, args.state_dim), device=device)

                input_t0 = base_inputs.clone()
                input_t0[args.observer_agent] = obs_encoded[0]
                prev_neighbors = gather_neighbor_states(prev_states, neighbor_idx_t, neighbor_mask_t)
                curr_states = shared_agent.form_states(input_t0, prev_neighbors, agent_ids_t)

                loss = torch.tensor(0.0, device=device)
                step_metrics = []

                for t in range(args.unroll_steps):
                    next_input = base_inputs.clone()
                    next_input[args.observer_agent] = obs_encoded[t + 1]

                    curr_neighbors = gather_neighbor_states(curr_states, neighbor_idx_t, neighbor_mask_t)
                    next_states = shared_agent.form_states(next_input, curr_neighbors, agent_ids_t)

                    obs_state = curr_states[args.observer_agent : args.observer_agent + 1]
                    obs_neighbors = curr_neighbors[args.observer_agent : args.observer_agent + 1]
                    obs_logits = shared_agent.predict_next_token_logits(obs_state, obs_neighbors, observer_id_t).squeeze(0)
                    ce = F.cross_entropy(obs_logits, tokens[:, t + 1], reduction="mean")
                    loss = loss + ce
                    step_metrics.append((True, float(ce.detach().item())))

                    pred_neighbors = shared_agent.predict_neighbor_states(curr_states, curr_neighbors, agent_ids_t)
                    target_neighbors = gather_neighbor_states(next_states, neighbor_idx_t, neighbor_mask_t)
                    mse = masked_neighbor_mse(
                        pred_neighbors,
                        target_neighbors,
                        neighbor_mask_t,
                        batch_size=args.batch_size,
                        state_dim=args.state_dim,
                    )
                    loss = loss + mse
                    step_metrics.append((False, float(mse.detach().item())))

                    curr_states = next_states

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            for is_obs_metric, metric_val in step_metrics:
                total_metric += metric_val
                total_count += 1
                if is_obs_metric:
                    obs_ce += metric_val
                    obs_count += 1
                else:
                    neighbor_mse += metric_val
                    neighbor_count += 1

        if ep % args.log_interval == 0:
            total_avg = total_metric / max(1, total_count)
            obs_avg = obs_ce / max(1, obs_count)
            mse_avg = neighbor_mse / max(1, neighbor_count)
            if args.eval_batches > 0:
                eval_ce, eval_acc = autoregressive_token_eval()
                print(
                    f"Ep {ep:4d} | CE observer: {obs_avg:.4f} | "
                    f"MSE neighbor-state (all agents): {mse_avg:.4f} | mixed total: {total_avg:.4f} | "
                    f"AR eval CE: {eval_ce:.4f} | AR eval acc: {eval_acc:.4f}"
                )
            else:
                print(
                    f"Ep {ep:4d} | CE observer: {obs_avg:.4f} | "
                    f"MSE neighbor-state (all agents): {mse_avg:.4f} | mixed total: {total_avg:.4f}"
                )


if __name__ == "__main__":
    main()
