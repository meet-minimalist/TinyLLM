# End-to-end LLM training: notes and plan

Goal: train a ~3B-total / 300–500M-active model through pretraining, mid-training
and post-training, and keep the repo flexible for architecture experiments.

Reference: [Rigel Base](https://open-lm-engine.github.io/blog/rigel/)
(open-lm-engine, code: [lm-engine](https://github.com/open-lm-engine/lm-engine)).

---

## 1. What Rigel actually did

| Item | Rigel Base |
|---|---|
| Params | 2.3B total, 360M active (260M active non-embedding) |
| Layers | 40 layers, d_model 1024. 30 Mamba-2 + 10 GQA (attention every 4th layer) |
| Attention | GQA (group 4), **XSA on every attention layer**, NoPE (no RoPE), SWA 4096 in long-context phase |
| FFN | MoE in every layer: **128 SwiGLU experts, expert hidden size 128, top-2**, no shared expert, aux load-balance loss |
| Vocab | 100,352 |
| Init / HP transfer | μP |
| Optimizer | AdamW, peak LR 0.01 (μP units), 5,000 warmup steps |
| LR schedule | Power scheduler: `min(η_max, 1.819·s^-0.51)`, then linear decay to 0 |
| Batch | 1,152 × 4096 = 4.7M tokens/step, constant token count for the whole run |
| Compute | 5.4e21 non-embedding FLOPs ≈ **3.5T tokens** (5.4e21 / (6 × 2.6e8)) |
| Data | 5 pretraining phases: web+code → STEM swap → more math → Nemotron-CC-v2 in, math/code 35% each → Nemotron-CC dominant |
| Long context | Phase 6: 118B tokens, 294,912-token sequences, LR 5e-5, ring-attention context parallelism |
| Post-training | **Not in the blog.** Rigel Base is a base model. They list instruct models as future work. |

Parameter check (useful to learn how to count):

- One expert = 3 × 1024 × 128 = 393K params. 128 experts × 40 layers = **2.01B** (this is the "3B" part).
- Active FFN = 2 experts × 393K × 40 = **31M**. The MoE adds almost no active compute.
- Mamba-2 (expand 2) ≈ 6.6M/layer × 30 = ~200M. GQA ≈ 2.6M/layer × 10 = 26M.
- So most **active** compute is in the Mamba-2 mixers, and most **total** params are in experts.
- Tokens per active param ≈ 13,000. This is ~650× over Chinchilla. It is on purpose: small active size is cheap at inference.

Lessons they state: "work with the compute you have"; XSA beat GQA, gated attention and gated+XSA in their
ablation; deep (40 × 1024) beat wide for downstream scores; they chose Mamba-2 over Gated DeltaNet only
because of kernel cost on V100/TPU.

---

## 2. Scaling laws and rules of thumb

### 2.1 How many tokens

| Rule | Source | Use |
|---|---|---|
| Compute-optimal: D ≈ 20 N | Chinchilla, [2203.15556](https://arxiv.org/abs/2203.15556) | Only minimizes *training* compute. Good for proxy/ablation runs. |
| Inference-aware: train far past 20 N | [Beyond Chinchilla-Optimal 2401.00448](https://arxiv.org/abs/2401.00448), [Over-training 2403.08540](https://arxiv.org/abs/2403.08540) | Real small models use 1,000–10,000+ tokens/param (Llama-3 8B: 15T; Qwen3-0.6B: 36T; Rigel: ~13K per active param). Loss still falls predictably. |
| Training FLOPs ≈ 6 · N_active · D | Kaplan, [2001.08361](https://arxiv.org/abs/2001.08361) | For MoE use **active** non-embedding params. Add attention FLOPs for long context. |

Budget for your target (400M active non-embedding, ~250 TFLOP/s effective per H100 — realistic MFU for a small MoE):

| Tokens | FLOPs | H100-hours |
|---|---|---|
| 100B | 2.4e20 | ~270 |
| 500B | 1.2e21 | ~1,300 |
| 1T | 2.4e21 | ~2,700 |
| 2T | 4.8e21 | ~5,300 |
| 3.5T (Rigel) | 8.4e21 | ~9,300 |

### 2.2 Shape (depth, width, heads, FFN)

- Shape matters little over a wide range at fixed params (Kaplan). At small scale, **deep and thin wins**
  ([MobileLLM 2402.14905](https://arxiv.org/abs/2402.14905); Rigel 40 × 1024).
  Practical band for 0.3–0.5B active: d_model 1024–1536, 24–40 layers.
- head_dim 64–128. GQA with 4–8 query heads per KV head.
- Dense SwiGLU: hidden ≈ 8/3 · d (param-matched to 4× GELU); many models use 3–3.5×.
- QK-norm (OLMo 2, Qwen3) removes most loss spikes. Keep it.
- Vocab: bigger models want bigger vocab ([2407.13623](https://arxiv.org/abs/2407.13623)).
  At d=1024 a 100K–150K vocab costs 100–150M params. Tie embeddings at this size.

### 2.3 MoE knobs

Terms: E = experts, k = top-k, activation ratio = k/E, granularity G = (dense FFN hidden) / (expert hidden).

| Finding | Source |
|---|---|
| Fine-grained experts (many small experts, bigger k) beat few large experts. Optimal G grows with compute. | DeepSeekMoE [2401.06066](https://arxiv.org/abs/2401.06066), [Krajewski 2402.07871](https://arxiv.org/abs/2402.07871) |
| Efficiency Leverage (EL = dense compute matched / MoE compute) follows power laws in activation ratio and compute. Lower activation ratio helps more at larger compute. G ≈ 8–16 is a robust range. 0.85B active matched a 6.1B dense model (>7× leverage). At the **smallest** budgets, the sparsest setting was slightly worse. | Ling / Ant [2507.17702](https://arxiv.org/abs/2507.17702) |
| For a fixed compute budget, optimal sparsity grows as total params grow. | Abnar et al. (Apple) [2501.12370](https://arxiv.org/abs/2501.12370) |
| Load balance: aux loss (Switch / ST-MoE), router z-loss ([2202.08906](https://arxiv.org/abs/2202.08906)), aux-loss-free bias balancing ([DeepSeek-V3 2412.19437](https://arxiv.org/abs/2412.19437)), balance over the **global** batch not the micro-batch ([2501.11873](https://arxiv.org/abs/2501.11873)). | — |
| Shared expert: DeepSeek uses one; Qwen3 and Rigel drop it. Small effect; ablate. | — |

Rule of thumb for you: activation ratio 1/32–1/16 (e.g. 128 experts, top-4 to top-8), G ≈ 8–16,
MoE in every layer (or dense first layer), global-batch balancing, router in fp32.

### 2.4 Sequence mixers (attention and hybrids)

- Hybrids: Mamba-2 ([2405.21060](https://arxiv.org/abs/2405.21060)) or Gated DeltaNet
  ([2412.06464](https://arxiv.org/abs/2412.06464)) plus a few full-attention layers. Used in Granite 4-H,
  Nemotron-H, Qwen3-Next / Qwen3.5, Kimi Linear, Rigel.
- Linear:full ratio **3:1 to 6:1**. LM loss is flat across ratios; recall drops below ~3:1 full attention.
  Gated DeltaNet / HGRN-2 are the best linear partners ([2507.06457](https://arxiv.org/abs/2507.06457)).
- With recurrent layers, attention layers can use **NoPE**; position comes from the recurrence. This makes
  long-context extension easy (Rigel, Kimi Linear).
- Attention tweaks worth a flag: gated attention (sigmoid output gate, kills attention sinks,
  [2505.06708](https://arxiv.org/abs/2505.06708)), XSA ([2603.09078](https://arxiv.org/abs/2603.09078), already in this repo), SWA.

### 2.5 Hyperparameters

| Rule | Source |
|---|---|
| η_opt = 0.3118 · C^-0.125, B_opt = 0.292 · C^0.327 (B in tokens, C in FLOPs) | DeepSeek LLM [2401.02954](https://arxiv.org/abs/2401.02954) |
| η_opt = 1.79 · N^-0.713 · D^0.307, B_opt = 0.58 · D^0.571 (tokens) | Step Law [2503.04715](https://arxiv.org/abs/2503.04715) |
| Optimal and critical batch size depend on D, not on N. Keep AdamW timescale τ = B/(η·λ·D) at its optimum; optimal τ is a power law in tokens/param. So λ scales with B. | Power Lines [2505.13738](https://arxiv.org/abs/2505.13738) |
| μP: tune LR/init on a narrow model, transfer to wide. CompleteP extends this to depth. | [2203.03466](https://arxiv.org/abs/2203.03466) |
| Muon: scale update RMS to match AdamW (0.2·√max(m,n)), then AdamW LR and WD transfer. ~2× token efficiency claimed. | Moonlight [2502.16982](https://arxiv.org/abs/2502.16982) |

Worked example for 400M active, 2T tokens (C ≈ 4.8e21): DeepSeek gives B ≈ 3.6M tokens, η ≈ 6e-4 (AdamW,
standard param). Step Law gives B ≈ 6M tokens. Rigel used 4.7M. So plan for **~4M tokens/step**.

### 2.6 LR schedule

- **WSD** (warmup–stable–decay, [MiniCPM 2404.06395](https://arxiv.org/abs/2404.06395)): hold LR flat, decay
  in the last 10–20%. You can branch the decay from any stable checkpoint. Mid-training data goes into the decay.
- **Power scheduler** ([2408.13359](https://arxiv.org/abs/2408.13359)): what Rigel uses. Same schedule works
  for any batch size and token count, so you do not need to fix the total tokens in advance.
- Decay to **zero** linearly beats cosine-to-10% (Bergsma et al. 2025, "Straight to Zero").

---

## 3. Data

### 3.1 Pretraining sources (all public)

| Domain | Datasets |
|---|---|
| Web | FineWeb-Edu, DCLM-baseline, Nemotron-CC (v1, v2) |
| PDFs / long docs | FinePDFs |
| Code | Stack-Edu, The Stack v2 subsets |
| Math | FineMath, MegaMath, Nemotron-CC-Math |
| Multilingual | FineWeb-2 |

### 3.2 Phases (pattern shared by Rigel, SmolLM2/3, OLMo 2, Llama 3)

1. **Stable phase**: mostly web (~70–85%), some code and math.
2. **Shift phases**: swap in higher-quality web, raise math/code (Rigel reached 35% each).
3. **Mid-training / anneal** (the LR decay, 5–15% of tokens): best-quality web, math, code, plus
   instruction-style and reasoning QA data (OLMo 2 "Dolmino", SmolLM3 decay).
4. **Long-context extension**: long documents + long-CoT QA, low LR.

Mixture search: train many small proxy models on different mixes, then fit which mix wins
([RegMix 2407.01492](https://arxiv.org/abs/2407.01492), data mixing laws). Your 50M setup is a good proxy.

---

## 4. Post-training

Standard open recipe (Tülu 3 [2411.15124](https://arxiv.org/abs/2411.15124), OLMo 3, SmolLM3):

1. **SFT**: chat template, loss only on assistant tokens. Data: Tülu 3 SFT mix, SmolTalk2, OpenThoughts,
   Nemotron post-training sets.
2. **Preference tuning (DPO)**: Tülu 3 preferences. OLMo 3 "delta learning": chosen = strong model,
   rejected = weak model.
3. **RLVR**: RL with verifiable rewards (math answers, unit tests, instruction constraints). GRPO
   ([2402.03300](https://arxiv.org/abs/2402.03300)), DAPO ([2503.14476](https://arxiv.org/abs/2503.14476)),
   Dr. GRPO ([2503.20783](https://arxiv.org/abs/2503.20783)).

### 4.1 The terms, separated

Two questions: *where does the signal come from*, and *which algorithm uses it*.

| Method | Signal | Online? | Needs |
|---|---|---|---|
| SFT | good examples (copy them) | no | chat data, loss on assistant tokens only |
| DPO | pairs: chosen vs rejected | no | preference data + frozen reference (SFT) model, β limits drift |
| RLHF-PPO | learned reward model | yes | reward model, critic, generation |
| RLVR + GRPO | a program checks the answer (math, tests, constraints) | yes | verifiers, generation |
| On-policy distillation | teacher's probability for each token of the student's sample | yes | teacher model, generation |

- RLVR is a *reward source* (a checker). GRPO is an *algorithm*.
- GRPO: sample a group (e.g. 8) per prompt, reward each, advantage = (r − mean) / std within the group,
  push up above-average samples. No critic network. DAPO / Dr. GRPO fix clipping, zero-signal groups and
  length bias.
- DPO is offline (fixed data). GRPO and on-policy distillation are online (the model generates each step),
  so they need a fast generation engine.

### 4.2 Reasoning and code: distill from a big model

Yes, this is the main lever for a small model.

- DeepSeek-R1 distilled its traces into 1.5B–70B dense models, and the distilled small models beat RL
  run directly on them. Qwen3's small models come from strong-to-weak distillation.
- Put reasoning data in **mid-training** as well as SFT (OLMo 3, SmolLM3; Rigel's last phase is
  long-CoT QA). SFT/RL then start from a higher base.
- Reasoning traces are 4K–32K tokens, so long-context extension must come first. Small models loop on
  very long CoT: filter traces for correctness and cap their length.
- **Tokenizer consequence**: logit-level / on-policy distillation needs the student and teacher to
  share a tokenizer. Using the Qwen3 tokenizer (~151K) lets any Qwen3-family model be the teacher.
  Cost at d=1024: ~155M embedding params. This is the recommended P1 choice.
- Licence: check the teacher's output terms (e.g. Apache-2.0 / MIT models) before a public release.
- Sources: OpenThoughts, OpenR1-Math, NVIDIA OpenCodeReasoning / Nemotron post-training sets.
- Evals: GSM8K, MATH-500, HumanEval+/MBPP+, LiveCodeBench. Compare with Qwen3-0.6B/1.7B, SmolLM3-3B.

### 4.3 Data-mixing sweeps

[codelion, "optimal dataset mixing"](https://huggingface.co/blog/codelion/optimal-dataset-mixing): a
64M GPT-2 on 1B tokens; static 50% FinePDFs / 30% DCLM / 20% FineWeb-Edu beat curricula. Pure FinePDFs
had the best in-domain perplexity but 717× worse out-of-domain perplexity. Use: always track an
out-of-domain perplexity. Do not reuse the ratios: one small scale, perplexity only, no code/math,
and the multi-phase gains at scale come from the LR-decay phase, which this setup does not test.

For a 300–500M-active model: **distillation gives more than RL**. Qwen3 small models are distilled from big
ones. On-policy distillation (student samples, teacher scores per token with reverse KL,
[GKD 2306.13649](https://arxiv.org/abs/2306.13649)) is the 2026 default. Use RL last, and small.

---

## 5. Gap analysis: this repo today

| Needed | Status |
|---|---|
| Multi-GPU (DDP / FSDP2) | Missing. 3B total params needs ~48 GB for fp32 weights + Adam state alone. |
| Full checkpoint resume (model, optimizer, scheduler, data position, RNG) | Missing (see WORKLOG). Blocks every multi-phase plan. |
| Modern tokenizer, uint32 token shards | Missing. Only GPT-2 pre-tokenized FineWeb10B. |
| Multi-source weighted data mix per phase | Missing. |
| Heterogeneous layer pattern (e.g. `MMMA`) | Missing. All blocks are identical. |
| MoE FFN (fine-grained, top-k, balancing, grouped GEMM) | Missing. |
| Mamba-2 / Gated DeltaNet mixers | Missing (Linux kernels: `mamba-ssm`, `flash-linear-attention`). |
| NoPE, gated attention, SWA | Partial (XSA done, SWA possible via flash-attn). |
| WSD / power / linear-to-zero schedules | Missing (cosine, linear, constant only). |
| μP, sweep + power-law fitting tools | Missing (Chinchilla estimator only). |
| Active vs total param and FLOP counter | Missing. |
| SFT loss masking, chat template, DPO, RLVR / distillation | Missing. |
| HF export (for lm-eval-harness, vLLM) | Missing. |

Keep: the layer registry, config-driven builder, varlen packing with per-doc RoPE reset, Muon, the
analysis callbacks, XSA.

---

## 6. Decisions (2026-09-30)

- **Compute**: free compute first (Kaggle 2×T4, Colab, Lightning credits; TPU Research Cloud later via
  torch-xla). Rent (vast.ai) later. Code must run on 1 GPU and 8 GPUs; multi-node later.
  - T4/P100 have no bf16 and no flash-attn 2 → fp16 + GradScaler and SDPA fallback must work.
  - Sessions end after ~12 h → full, frequent, time-triggered resume is the top priority.
- **Tokenizer**: reuse an open tokenizer with chat tokens (dolma2 ~100K or Qwen3 ~151K).
- **Architecture**: attention MoE first (on GQA/XSA). Gated DeltaNet / Mamba-2 after.
- **Post-training**: SFT → DPO → on-policy distillation → small RLVR.
- **Purpose**: learning + a public result (HF release, report with one clean ablation question, reproducible code).

## 7. Proposed roadmap (one PR each)

| # | Stage | Content |
|---|---|---|
| P0 | Foundations (branch `feat/p0-foundations`) | **Done:** full resume (model, optimizer, scheduler, scaler, data position, RNG), time-limit / SIGTERM save, HF-hub checkpoint sync, `init_from`, fp16 + GradScaler for T4, DDP, WSD + power schedules, param/FLOP/MFU accounting. **Moved to P2:** FSDP2 (needed once the MoE model outgrows one GPU's memory). A `stage` config comes with P4/P5. |
| P1 | Data | Pick tokenizer; tokenize HF datasets to uint32 shards with a manifest; weighted multi-source loader; per-phase mixture YAML. |
| P2 | Architecture | `layer_pattern` in the model YAML; MoE FFN (top-k, shared expert flag, aux / aux-free balance, z-loss, `torch._grouped_mm`); NoPE; gated attention; SWA; Mamba-2 and Gated DeltaNet via libraries. |
| P3 | Scaling toolkit | μP; sweep runner over width/LR/batch; fit power laws; isoFLOP plots. |
| P4 | Mid-training + long context | Anneal phase configs; RoPE-θ / YaRN or NoPE+SWA extension. |
| P5 | Post-training | SFT (chat template + loss mask) → DPO → on-policy distillation → small RLVR. HF export. |
| P6 | Evals | lm-eval-harness via HF export; generative evals (GSM8K, IFEval, HumanEval) for post-trained models. |

Learning ladder before the big run:

1. 50M dense (today) → fix mixture and LR on proxies.
2. ~150M-total / 30M-active MoE → check balancing, EL vs dense.
3. ~600M-total / 100M-active → fit LR/batch scaling, choose hybrid ratio.
4. 3B-total / 400M-active → final run.

A first candidate for the final model (attention-only MoE; to be checked by the ladder):
d_model 1024, 28 layers, GQA 16 q / 4 kv heads, head_dim 64, 128 experts of hidden 256, top-8
(activation ratio 1/16, G = 16), 100K vocab tied → ~3.0B total, ~355M active.
A Rigel-style hybrid (3:1 GDN or Mamba-2 : attention, NoPE) is the second candidate.
