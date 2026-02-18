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


class Agent(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        state_dim: int,
        num_neighbors: int,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_neighbors = num_neighbors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = MLP(embed_dim, state_dim, hidden_dim, hidden_layers)
        pred_in = state_dim * (1 + num_neighbors)
        self.state_updater = MLP(pred_in, state_dim, hidden_dim, hidden_layers)
        self.token_predictor = MLP(pred_in, vocab_size, hidden_dim, hidden_layers)
        self.state_predictor = MLP(pred_in, state_dim * num_neighbors, hidden_dim, hidden_layers)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(token_ids)
        return self.encoder(emb)

    def _features(self, self_state: torch.Tensor, neighbor_states: torch.Tensor) -> torch.Tensor:
        # self_state: [B, state_dim], neighbor_states: [B, num_neighbors, state_dim]
        bsz = self_state.shape[0]
        flat_neighbors = neighbor_states.reshape(bsz, -1)
        return torch.cat([self_state, flat_neighbors], dim=-1)

    def form_state(self, input_state: torch.Tensor, prev_neighbor_states: torch.Tensor) -> torch.Tensor:
        return self.state_updater(self._features(input_state, prev_neighbor_states))

    def predict_next_token(self, self_state: torch.Tensor, neighbor_states: torch.Tensor) -> torch.Tensor:
        return self.token_predictor(self._features(self_state, neighbor_states))  # [B, vocab]

    def predict_neighbor_states(self, self_state: torch.Tensor, neighbor_states: torch.Tensor) -> torch.Tensor:
        pred = self.state_predictor(self._features(self_state, neighbor_states))
        return pred.reshape(self_state.shape[0], self.num_neighbors, self.state_dim)


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

    agents = nn.ModuleList(
        [
            Agent(
                vocab_size=vocab_size,
                embed_dim=args.embed_dim,
                state_dim=args.state_dim,
                num_neighbors=len(graph.neighbors[i]),
                hidden_dim=args.hidden_dim,
                hidden_layers=args.hidden_layers,
            )
            for i in range(args.agents)
        ]
    ).to(device)

    opt = torch.optim.Adam(agents.parameters(), lr=args.lr)
    total_params = sum(p.numel() for p in agents.parameters())
    trainable_params = sum(p.numel() for p in agents.parameters() if p.requires_grad)
    per_agent_params = sum(p.numel() for p in agents[0].parameters()) if args.agents > 0 else 0
    neighbor_counts = [len(graph.neighbors[i]) for i in range(args.agents)]
    neighbor_index_tensors = [
        torch.as_tensor(graph.neighbors[i], dtype=torch.long, device=device) for i in range(args.agents)
    ]

    print("Vocab size:", vocab_size)
    print("Agents:", args.agents)
    print("Observer agent:", args.observer_agent)
    print("Graph:", args.graph)
    if args.graph not in {"full", "ring", "line", "grid", "star"}:
        print("Neighbors per agent:", k_neighbors)
    print("Unroll steps:", args.unroll_steps)
    print("Device:", device)
    print("Model overview:")
    print("  Single observer receives token; all agents receive previous-step neighbor states")
    print("  All agents -> neighbor-next-state MSE; observer additionally -> next-token CE")
    print(
        "  Observation can be viewed as an external node connected only to the observer through its token input"
    )
    print(f"  Trainable parameters: {trainable_params:,} / total: {total_params:,}")
    print(f"  Parameters per agent (agent 0): {per_agent_params:,}")
    print(
        "  Neighbor count stats (min/avg/max): "
        f"{min(neighbor_counts)}/{(sum(neighbor_counts) / len(neighbor_counts)):.2f}/{max(neighbor_counts)}"
    )

    def update_states(input_states: torch.Tensor, prev_states: torch.Tensor) -> torch.Tensor:
        # input_states: [A, B, state_dim], prev_states: [A, B, state_dim]
        out_states = []
        for i in range(args.agents):
            neigh_idx = neighbor_index_tensors[i]
            prev_neigh_states = torch.index_select(prev_states, 0, neigh_idx).permute(1, 0, 2)
            s_i = agents[i].form_state(input_states[i], prev_neigh_states)
            out_states.append(s_i)
        return torch.stack(out_states, dim=0)

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

            zero_embed = torch.zeros((args.batch_size, args.embed_dim), device=device)
            zero_input_states = []
            for i in range(args.agents):
                zero_input_states.append(agents[i].encoder(zero_embed))
            zero_input_states_t = torch.stack(zero_input_states, dim=0)  # [A, B, state_dim]

            prev_states = torch.zeros((args.agents, args.batch_size, args.state_dim), device=device)
            loss = torch.tensor(0.0, device=device)
            step_metrics = []

            # Build t=0 states from token_0 and zero previous states.
            input_states_0 = []
            for i in range(args.agents):
                if i == args.observer_agent:
                    input_states_0.append(agents[i].encode(tokens[:, 0]))
                else:
                    input_states_0.append(zero_input_states_t[i])
            curr_states = update_states(torch.stack(input_states_0, dim=0), prev_states)

            # Predict transitions for t -> t+1, where t in [0, unroll_steps-1].
            for t in range(args.unroll_steps):
                next_input_states = []
                for i in range(args.agents):
                    if i == args.observer_agent:
                        next_input_states.append(agents[i].encode(tokens[:, t + 1]))
                    else:
                        next_input_states.append(zero_input_states_t[i])
                next_states = update_states(torch.stack(next_input_states, dim=0), curr_states)

                obs_idx = args.observer_agent
                obs_neigh_idx = neighbor_index_tensors[obs_idx]
                obs_neigh_states = torch.index_select(curr_states, 0, obs_neigh_idx).permute(1, 0, 2)
                logits = agents[obs_idx].predict_next_token(curr_states[obs_idx], obs_neigh_states)
                ce = F.cross_entropy(logits, tokens[:, t + 1], reduction="mean")
                loss = loss + ce
                step_metrics.append((True, ce.item()))

                for i in range(args.agents):
                    neigh_idx = neighbor_index_tensors[i]
                    neigh_states = torch.index_select(curr_states, 0, neigh_idx).permute(1, 0, 2)
                    pred_neighbor_states = agents[i].predict_neighbor_states(curr_states[i], neigh_states)
                    target_neighbor_states = torch.index_select(next_states, 0, neigh_idx).permute(1, 0, 2)
                    mse = F.mse_loss(pred_neighbor_states, target_neighbor_states, reduction="mean")
                    loss = loss + mse
                    step_metrics.append((False, mse.item()))

                curr_states = next_states

            opt.zero_grad()
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
            print(
                f"Ep {ep:4d} | CE observer: {obs_avg:.4f} | "
                f"MSE neighbor-state (all agents): {mse_avg:.4f} | mixed total: {total_avg:.4f}"
            )


if __name__ == "__main__":
    main()
