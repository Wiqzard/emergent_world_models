# Language Experiment: Exact Mathematical Specification (Causal Rollout)

This document formalizes what `language_experiment.py` now does after removing future-token leakage.

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
- Encoder $f_i: \mathbb{R}^{d_e} \to \mathbb{R}^{d_s}$ (MLP)
- Token head $g_i^{\text{tok}}: \mathbb{R}^{d_s(1+N_i)} \to \mathbb{R}^{V}$ (MLP)
- Neighbor-state head $g_i^{\text{st}}: \mathbb{R}^{d_s(1+N_i)} \to \mathbb{R}^{N_i d_s}$ (MLP)

## 2. Data Construction

Sentences are tokenized, lowercased, and wrapped with BOS/EOS:

$$
\text{ids}(s) = [\texttt{<bos>}, w_1, \dots, w_L, \texttt{<eos>}].
$$

For each batch element $b$, sample a window of length $A+1$:

$$
\mathbf{w}^{(b)} = [w_0^{(b)}, w_1^{(b)}, \dots, w_A^{(b)}].
$$

Then define aligned matrices:

$$
X_t[b,i] = w_i^{(b)}, \qquad X_{t+1}[b,i] = w_{i+1}^{(b)}.
$$

## 3. Causal Visibility Schedule

At rollout step $k$, observer agent $i$ sees its token iff $i \le k$.

Define visibility indicator:

$$
v_i^{(k)} = m_i \cdot \mathbf{1}[i \le k].
$$

So:

- Step $0$: only observer agent $0$ can use token input.
- Step $1$: observer agents $0,1$ can use token input.
- ...
- Step $k$: observer agents $\{0,\dots,k\}$ can use token input.

This is prefix reveal over agent index.

## 4. State Encoding at Step $k$

For each agent $i$ and sample $b$:

If visible observer ($v_i^{(k)}=1$):

$$
s_{i,k}^{(b)} = f_i(E_i[X_t[b,i]]), \qquad s^{\text{next},(b)}_{i,k} = f_i(E_i[X_{t+1}[b,i]]).
$$

Otherwise (non-observer or not-yet-revealed observer), with $\mathbf{0} \in \mathbb{R}^{d_e}$:

$$
s_{i,k}^{(b)} = f_i(\mathbf{0}), \qquad s^{\text{next},(b)}_{i,k} = f_i(\mathbf{0}).
$$

## 5. Feature Assembly

For each agent $i$ at step $k$:

$$
\mathbf{n}_{i,k}^{(b)} = \operatorname{concat}\left(\{s_{j,k}^{(b)} : j \in \mathcal{N}(i)\}\right) \in \mathbb{R}^{N_i d_s}
$$

$$
\mathbf{h}_{i,k}^{(b)} = \operatorname{concat}(s_{i,k}^{(b)}, \mathbf{n}_{i,k}^{(b)}) \in \mathbb{R}^{d_s(1+N_i)}.
$$

## 6. Outputs and Losses

### 6.1 Token prediction (only active agent $k$)

If agent $k$ is observer ($m_k=1$):

$$
\ell_k^{(b)} = g_k^{\text{tok}}(\mathbf{h}_{k,k}^{(b)}) \in \mathbb{R}^{V},
$$

with CE loss:

$$
\mathcal{L}_{k}^{\text{obs}} =
-\frac{1}{B}\sum_{b=1}^{B}
\log \operatorname{softmax}(\ell_k^{(b)})_{X_{t+1}[b,k]}.
$$

If $m_k=0$, no token loss is added at that step.

### 6.2 Neighbor-state prediction (all non-observers, every step)

For each non-observer $i$ ($m_i=0$):

$$
\hat{\mathbf{S}}_{i,k}^{(b)} = \operatorname{reshape}\!\left(g_i^{\text{st}}(\mathbf{h}_{i,k}^{(b)}), (N_i,d_s)\right),
$$

with target:

$$
\mathbf{S}^{\text{next},(b)}_{i,k} = [\, s^{\text{next},(b)}_{j,k} \,]_{j \in \mathcal{N}(i)}.
$$

Per-step non-observer loss:

$$
\mathcal{L}_{k}^{\text{non}} =
\sum_{i: m_i=0}
\frac{1}{B N_i d_s}
\sum_{b=1}^{B}
\left\lVert
\hat{\mathbf{S}}_{i,k}^{(b)} - \mathbf{S}^{\text{next},(b)}_{i,k}
\right\rVert_2^2.
$$

### 6.3 Total optimization objective per sampled window

$$
\mathcal{L}_{\text{window}} = \sum_{k=0}^{A-1}
\left(
\mathbf{1}[m_k=1]\,\mathcal{L}_{k}^{\text{obs}} + \mathcal{L}_{k}^{\text{non}}
\right).
$$

Adam updates all agent parameters using this summed loss.

## 7. Why This Removes Future-Token Leakage

At step $k$, agent $k$ predicts target token:

$$
y_k^{(b)} = X_{t+1}[b,k] = w_{k+1}^{(b)}.
$$

Potential leakage would require a neighbor $j$ carrying representation of $w_{k+1}^{(b)}$ in input features to agent $k$.

The only agent aligned to token $w_{k+1}$ is agent $j=k+1$. But at step $k$:

$$
v_{k+1}^{(k)} = m_{k+1}\cdot \mathbf{1}[k+1 \le k] = 0,
$$

so agent $k+1$ is not revealed yet and contributes only a zero-input state, not a token-conditioned state.

Therefore, for the token objective at step $k$, direct access to the target token through neighbor states is removed.
