# Language Experiment: Exact Mathematical Specification (Causal + Recurrent Neighbor Update)

This document formalizes the current behavior of `language_experiment.py` and `language_experiment_efficient.py`.

## 1. Notation

- Number of agents: $A$
- Batch size: $B$
- Vocabulary size: $V$
- Embedding dimension: $d_e$
- Latent state dimension: $d_s$
- Agent index: $i \in \{0,\dots,A-1\}$
- Rollout step index: $k \in \{0,\dots,A-1\}$
- Neighbor set of agent $i$: $\mathcal{N}(i)$ with size $N_i = |\mathcal{N}(i)|$
- Observer mask: $m_i \in \{0,1\}$ where $m_i=1$ means observer

Each agent $i$ has parameters:

- Embedding table $E_i \in \mathbb{R}^{V \times d_e}$
- Input encoder $f_i: \mathbb{R}^{d_e} \to \mathbb{R}^{d_s}$
- State updater $u_i: \mathbb{R}^{d_s(1+N_i)} \to \mathbb{R}^{d_s}$
- Token head $g_i^{\text{tok}}: \mathbb{R}^{d_s(1+N_i)} \to \mathbb{R}^{V}$
- Neighbor-state head $g_i^{\text{st}}: \mathbb{R}^{d_s(1+N_i)} \to \mathbb{R}^{N_i d_s}$

(All are MLPs except the embedding table.)

## 2. Data Construction

Sentences are tokenized, lowercased, and wrapped with BOS/EOS:

$$
\text{ids}(s) = [\texttt{<bos>}, w_1, \dots, w_L, \texttt{<eos>}].
$$

For each batch element $b$, sample a window of length $A+1$:

$$
\mathbf{w}^{(b)} = [w_0^{(b)}, w_1^{(b)}, \dots, w_A^{(b)}].
$$

Define aligned matrices:

$$
X_t[b,i] = w_i^{(b)}, \qquad X_{t+1}[b,i] = w_{i+1}^{(b)}.
$$

## 3. Causal Visibility Schedule

At rollout step $k$, observer agent $i$ sees its token iff $i \le k$:

$$
v_i^{(k)} = m_i \cdot \mathbf{1}[i \le k].
$$

So reveal is prefix-by-index: step $0$ reveals observer $0$, step $1$ reveals observers $0,1$, etc.

## 4. Per-Step Input Encodings

For stream $X_t$:

$$
\tilde{s}_{i,k}^{(b)} =
\begin{cases}
f_i(E_i[X_t[b,i]]) & \text{if } v_i^{(k)}=1 \\
f_i(\mathbf{0}) & \text{otherwise}
\end{cases}
$$

For stream $X_{t+1}$ (used as latent target stream):

$$
\tilde{s}^{+,(b)}_{i,k} =
\begin{cases}
f_i(E_i[X_{t+1}[b,i]]) & \text{if } v_i^{(k)}=1 \\
f_i(\mathbf{0}) & \text{otherwise}
\end{cases}
$$

with $\mathbf{0} \in \mathbb{R}^{d_e}$.

## 5. Recurrent State Update From Previous Neighbor States

Initialize previous-step recurrent states:

$$
s_{i,-1}^{(b)} = \mathbf{0}, \qquad s_{i,-1}^{+,(b)} = \mathbf{0}.
$$

For each rollout step $k$, define neighbor aggregates from **previous** step:

$$
\mathbf{n}_{i,k-1}^{(b)} = \operatorname{concat}(\{s_{j,k-1}^{(b)} : j \in \mathcal{N}(i)\}),
$$

$$
\mathbf{n}_{i,k-1}^{+,(b)} = \operatorname{concat}(\{s_{j,k-1}^{+,(b)} : j \in \mathcal{N}(i)\}).
$$

Then update current recurrent states:

$$
s_{i,k}^{(b)} = u_i\!\left(\operatorname{concat}(\tilde{s}_{i,k}^{(b)}, \mathbf{n}_{i,k-1}^{(b)})\right),
$$

$$
s_{i,k}^{+,(b)} = u_i\!\left(\operatorname{concat}(\tilde{s}_{i,k}^{+,(b)}, \mathbf{n}_{i,k-1}^{+,(b)})\right).
$$

So each new state uses current local input plus neighbor states from the previous rollout step.

## 6. Outputs and Losses

### 6.1 Token prediction (only active agent $k$)

If agent $k$ is observer ($m_k=1$):

$$
\mathbf{h}_{k,k}^{(b)} = \operatorname{concat}(s_{k,k}^{(b)}, \operatorname{concat}(\{s_{j,k}^{(b)}: j \in \mathcal{N}(k)\})),
$$

$$
\ell_k^{(b)} = g_k^{\text{tok}}(\mathbf{h}_{k,k}^{(b)}) \in \mathbb{R}^{V},
$$

$$
\mathcal{L}_{k}^{\text{obs}} =
-\frac{1}{B}\sum_{b=1}^{B}
\log \operatorname{softmax}(\ell_k^{(b)})_{X_{t+1}[b,k]}.
$$

If $m_k=0$, no token CE term is added at that step.

### 6.2 Neighbor-state prediction (all non-observers, every step)

For each non-observer $i$ ($m_i=0$):

$$
\mathbf{h}_{i,k}^{(b)} = \operatorname{concat}(s_{i,k}^{(b)}, \operatorname{concat}(\{s_{j,k}^{(b)}: j \in \mathcal{N}(i)\})),
$$

$$
\hat{\mathbf{S}}_{i,k}^{(b)} = \operatorname{reshape}(g_i^{\text{st}}(\mathbf{h}_{i,k}^{(b)}), (N_i,d_s)),
$$

$$
\mathbf{S}_{i,k}^{+,(b)} = [\, s_{j,k}^{+,(b)} \,]_{j \in \mathcal{N}(i)}.
$$

Per-step non-observer loss:

$$
\mathcal{L}_{k}^{\text{non}} =
\sum_{i: m_i=0}
\frac{1}{B N_i d_s}
\sum_{b=1}^{B}
\left\lVert
\hat{\mathbf{S}}_{i,k}^{(b)} - \mathbf{S}_{i,k}^{+,(b)}
\right\rVert_2^2.
$$

### 6.3 Total objective for one sampled window

$$
\mathcal{L}_{\text{window}} = \sum_{k=0}^{A-1}
\left(
\mathbf{1}[m_k=1] \, \mathcal{L}_{k}^{\text{obs}} + \mathcal{L}_{k}^{\text{non}}
\right).
$$

Adam updates all agent parameters using this summed loss.

## 7. Leakage Property Under This Schedule

At step $k$, token target is $y_k^{(b)} = X_{t+1}[b,k] = w_{k+1}^{(b)}$.

The only agent aligned to token $w_{k+1}$ is agent $k+1$, but at step $k$:

$$
v_{k+1}^{(k)} = m_{k+1}\cdot\mathbf{1}[k+1 \le k] = 0.
$$

So the current-step state update cannot inject token-conditioned information from agent $k+1$ at step $k$. This removes direct same-step target-token leakage through neighbor features for the token prediction at step $k$.
