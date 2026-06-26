# Training Metrics Interpretation Guide

All metrics are logged to WandB via `AnalysisCallback` every `every_n_steps` steps.
Enable sections individually in the training YAML under the `analysis` key.

---

## 1. Convergence (`train_loss`, `train_ppl`, `lr`)

Logged every step by the trainer.

| Metric | What it measures | Healthy | Warning |
|---|---|---|---|
| `train_loss` | Cross-entropy on training tokens | Smooth logarithmic decay | Flatlines or spikes to NaN |
| `train_ppl` | `e^loss` — model uncertainty | Decreasing steadily | PPL above 1000 after warmup suggests bad data or LR |
| `lr` | Active learning rate | Ramps up linearly, then decays | Stuck at peak or mismatched with step counter |

---

## 2. Gradient Health (`gradients/`)

Captured via backward hooks in `GradientCapture`. Hooks fire every backward pass; values are
collected and aggregated at analysis steps.

### Global scalars

| Metric | Formula | Healthy | Warning |
|---|---|---|---|
| `gradients/global/gradient_norm` | `sqrt(sum(‖g_i‖²))` | Stable after warmup, minor oscillations | Exponential hockey-stick → explosion. Sudden drop to 0 → vanishing |
| `gradients/global/exploded_gradients` | Count of params with norm > 1e4 | 0 | Any non-zero value is a red flag |
| `gradients/global/zero_gradients` | Count of params with norm = 0 | 0 | Dead parameters, possible dying ReLU or bad init |
| `gradients/global/snr` | `mean(‖EMA_mean(g)‖ / sqrt(EMA_var(g)))` | High during warmup (>1), then moderately positive | Near 0 → batch too noisy, LR too high, or model has run out of signal to learn |

**Gradient SNR interpretation:**
The SNR is approximated using an exponential moving average (β=0.99) of gradient
mean and variance across training steps. It measures directional certainty:
- **High SNR (>1):** Most gradient steps agree on direction — the model is making
  confident structural updates. Common in early warmup.
- **SNR near 1:** Normal convergence territory — the optimizer is exploring.
- **SNR near 0:** Gradient noise dominates signal. Usually means: LR is too high,
  batch size too small, or training has converged and there is nothing more to learn.

### Histograms (`gradients/hist/{param_name}`)

Distribution of gradient values for each parameter. Look for:
- **Concentrated near zero, light tails:** Healthy, well-regularized gradients.
- **Bimodal or multi-modal:** Can indicate conflicting gradients from different data patterns.
- **Extremely heavy tails:** Precursor to gradient explosion even if the norm is still OK.

---

## 3. Weight Stats (`weights/`)

Computed every analysis step directly from parameter values.

### SNR (`weights/snr/{param_name}`)

`SNR = mean(|W|) / std(|W|)`

Measures the ratio of the average magnitude to its spread. Unlike gradient SNR this is
static — it reflects the current state of the weights, not their update direction.

| Phase | Expected SNR | Interpretation |
|---|---|---|
| Early training (random init) | Low (~0.5–1.5) | Weights are noise — mean ≈ std by design of Xavier/Kaiming init |
| Mid training (convergence) | Steadily rising | **Feature crystallization.** Critical weights grow; uninformative ones shrink toward zero. This is healthy. |
| Late training | Plateaus | Normal — model has found its representational geometry |
| Spiking while val loss stalls | Very high spike | **Overfitting / memorization.** A few weight paths are growing disproportionately large. |

### Histograms (`weights/hist/{param_name}`)

Distribution of raw weight values. Look for:
- **Symmetric bell curve centered at 0:** Normal, well-initialized layer.
- **Distribution shifting and narrowing:** The layer is specializing — weights polarizing
  toward positive/negative extremes indicate strong feature selectivity.
- **Collapsing toward 0:** Layer may be dying or being suppressed by regularization.
- **Bimodal:** Common in late-layer MLPs — a sign of learned binary feature detectors.

---

## 4. Spectral Analysis (`spectral/`)

Computed via SVD on every 2D parameter. All metrics are **rotation-invariant** (they depend
only on singular values, not the arbitrary coordinate system of the weight matrix).

Enable with `spectral: true` in the analysis config. Note: SVD is expensive; use
`every_n_steps` of at least 500 for large models.

### Condition Number (`spectral/{name}/condition_number`)

`κ = σ_max / σ_min`

Measures how ill-conditioned the weight matrix is.

| κ value | Interpretation |
|---|---|
| ~1 | **Isotropic.** All singular values are similar — the layer treats all input directions equally. Typical of random init. |
| 10–100 | **Moderate specialization.** The layer has developed preferred input directions. Normal during learning. |
| 1,000–10,000 | **High specialization / near-singular.** Some directions are nearly suppressed. Common in late-layer attention projections. |
| > 1e6 or inf | **Rank-deficient.** The matrix has effectively zero singular values — a sign of rank collapse or vanishing gradients killing entire subspaces. |

### Stable Rank (`spectral/{name}/stable_rank`)

`stable_rank = ‖W‖_F² / ‖W‖_2² = sum(σ²) / σ_max²`

A soft, continuous version of matrix rank. Unlike algebraic rank, it doesn't collapse to
`min(M, N)` due to floating-point noise.

| Trend | Interpretation |
|---|---|
| High and stable | The matrix uses many dimensions — broad representational capacity. |
| Gradually decreasing | **Healthy compression.** The layer is discarding redundant directions and focusing on the most predictive ones. |
| Sudden sharp drop | **Rank collapse warning.** The layer is losing structural diversity. Watch for this alongside rising condition number. |

### Effective Rank (`spectral/{name}/effective_rank`)

`effective_rank = e^H(W)` where `H(W) = -sum(p_i * log(p_i))` and `p_i = σ_i / sum(σ_j)`

Treats the normalized singular value spectrum as a probability distribution and computes
its Shannon entropy. More sensitive to rank collapse than stable rank because entropy
penalizes any deviation from uniform distribution.

| Value | Interpretation |
|---|---|
| Near `min(M, N)` | **Maximum entropy** — singular values are all equal, like a random matrix. No structure learned yet. |
| Gradually decreasing | Normal learning — the layer is concentrating energy into fewer dominant directions. |
| Rapidly collapsing to 1–5 | **Representation collapse.** The layer is squeezing all information into 1–5 principal directions, discarding nearly all capacity. |

### Top-10 Energy Ratio (`spectral/{name}/top10_energy_ratio`)

`top10_energy_ratio = sum(σ²[:10]) / sum(σ²)`

What fraction of the total matrix energy is concentrated in the 10 largest singular values.
This is the permutation-invariant analogue of a "frequency-domain" check: low-frequency
structure corresponds to energy concentrated in a few dominant components.

| Value | Interpretation |
|---|---|
| Near `10 / min(M,N)` (e.g., ~0.01 for a 1024×1024 matrix) | **Flat spectrum** — energy spread uniformly, like random noise. No structure learned. |
| Steadily rising (e.g., 0.3–0.7) | **Healthy learning.** Global, low-frequency structure is forming in the matrix. |
| > 0.9 early in training | **Rank collapse warning.** Almost all energy is in 10 directions before the model has had a chance to learn complex representations. |
| > 0.9 late in training for small layers | May be acceptable — small embedding or head projection layers are expected to be low-rank. |

### Components for 95%/99% Variance (`spectral/{name}/n_for_95pct`, `n_for_99pct`)

The number of singular values needed to capture 95% or 99% of the matrix's total variance.
This directly answers "what is the effective dimensionality of this layer?"

- If a 1024×1024 matrix only needs 100 components for 99% variance, it is functionally
  operating in a 100-dimensional subspace.
- A large gap between `n_for_95pct` and `n_for_99pct` indicates a long tail of small
  singular values that contribute little individually but add up.

### WeightWatcher Alpha (`spectral/{name}/alpha`)

Power-law exponent fitted to the tail of the eigenvalue distribution `P(λ) ~ λ^(-α)`.

| α range | Interpretation |
|---|---|
| < 2 | **Heavy-tailed / over-specialized.** Very low α (< 1.5) can indicate memorization. |
| 2–4 | **Well-trained with good generalization.** Optimal range. |
| 4–6 | **Moderate structure.** Layer is learning but could benefit from more training. |
| 6–10 | **Near random init.** Limited learning has occurred in this layer. |
| → ∞ | **White noise.** All eigenvalues equal — layer is untrained. |

**Combined signal — what to look for:**

| `stable_rank` | `effective_rank` | `weight_snr` | Diagnosis |
|---|---|---|---|
| Dropping | Dropping | Dropping | **Representation collapse** — layer is turning into noise |
| Dropping | Dropping | Rising | **Healthy compression** — layer is specializing, pruning redundant directions |
| Stable | Stable | Rising slowly | **Normal convergence** |
| Stable | Stable | Spiking while val loss stalls | **Overfitting** |

---

## 5. Layer Cosine Similarity (`layer_cosim/`)

Measured using forward hooks on `TransformerBlock` modules.

### Per-pair scalars (`layer_cosim/layer_{i}_to_{i+1}`)

Cosine similarity between the output of block `i` and block `i+1`, averaged over all
token positions and batch elements.

| Value | Interpretation |
|---|---|
| Near 1.0 | Block `i+1` barely modifies the residual stream — near-identity transformation. Can indicate an under-trained or passthrough block. |
| 0.6–0.9 | Healthy range — the block is making meaningful but not radical changes to the representation. |
| < 0.5 | The block applies a large transformation. Can be healthy for early layers that are learning rapidly. |
| Decreasing over training | Blocks are progressively "activating" and contributing more to the representation. |
| Uniformly near 1.0 across all layers late in training | **Rank deficiency / representation collapse** — later layers are not differentiating the residual stream. |

### Table (`layer_cosim/table`)

WandB table with columns `[step, layer_0_to_1, layer_1_to_2, ...]`. Use WandB's table
chart builder to plot all layer pairs as overlaid time-series lines to see how each block's
contribution evolves over training.

---

## 6. Activation Histograms (`activations/hist/`)

Captured from attention layers via forward hooks. Only active when `track_activations: true`.
Tensors captured per attention layer: `q`, `k`, `v`, `qkv_out`, `o_proj_out`.

| Pattern | Interpretation |
|---|---|
| Symmetric, bell-shaped | Normal — activations are well-distributed. |
| Collapsing toward zero | Layer is under-activating. Possible vanishing gradient or dead attention heads. |
| Extremely heavy tails / saturation | Activations are saturating — consider reducing LR or adding gradient clipping. |
| Bimodal for `q`/`k` | Heads have specialized into distinct query/key regimes — often healthy in mid-late training. |

---

## 7. System (`system/`)

| Metric | What it measures | Notes |
|---|---|---|
| `system/gpu_memory_allocated_gb` | Bytes actively used by tensors | Track to stay below device limit |
| `system/gpu_memory_reserved_gb` | Bytes held by PyTorch's caching allocator | Always >= allocated. Spikes indicate fragmentation. |

---

## Quick Reference: Failure Mode Checklist

| Symptom | Metrics to check | Likely cause |
|---|---|---|
| Loss spikes to NaN | `gradient_norm`, `exploded_gradients` | Gradient explosion — reduce LR or tighten grad clip |
| Loss flatlines early | `gradient_snr`, `layer_cosim` | LR too high (SNR→0) or model not deep enough for the task |
| Val loss diverges from train loss | `weight_snr` spike, `alpha` < 2 | Overfitting / memorization |
| Loss decreases but very slowly | `alpha` 6–10, `effective_rank` near max | Under-training — increase LR or train longer |
| GPU OOM mid-run | `gpu_memory_reserved_gb` | Reduce batch size or enable activation checkpointing |
| Several layers show `layer_cosim` near 1.0 | `effective_rank` dropping | Representation collapse in those blocks — check LR and weight decay |
