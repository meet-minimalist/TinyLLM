# Work Log

A running log of the work to train a tiny LLM from scratch. Newest entry first.

---

## 2026-09-09 (later) — The LR fix worked, and the packer was throwing away 17% of the corpus

Ran the full epoch with `muon_lr: 0.02` (W&B run `hxjleaq0`). It works. Loss reached **3.589,
perplexity 36.2**, against 5.94 / PPL 380 on the old run. The non-finite guard never fired and there
was no NaN anywhere in the log.

But the run stopped at step 25345 of 30446 and went straight to evals. It did not stop early — the
dataloader ran out of data.

`setup.sh` downloads 10 shards at 100M tokens each, so 1.0B tokens were on disk. The run consumed
25345 x 32768 = 830.5M, which is 83.1%. The missing 17% was being discarded by the packer.

The cause was one line in `_iter_varlen`. `end` is clamped by three things — the next document,
`max_seq_len`, and the space left in the batch — and then `idx += 1` moved to the next document
unconditionally. So whenever either of the last two clamps bound, the rest of that document was
skipped and never read again. A 6,000-token document contributed 2,048 tokens and lost 3,952. The
last document of every batch lost its tail too.

Simulating the old loop against realistic FineWeb document lengths predicted 20-25% loss. The run
lost 16.9%, and the 2048 cap accounted for roughly six-sevenths of it.

The knock-on effect mattered more than the lost data. `num_training_steps` is Chinchilla-derived at
30446, so the cosine schedule was built for 30446 steps but ended at 25345. **The LR never finished
decaying** — the last step logged `LR: 0.001496`, not ~0. The model never got the end-of-cosine
anneal, so 3.589 is not the number this run would have produced with a completed schedule.

**Fix:** track an offset into the current document and only advance to the next one when the current
document is fully consumed. A long document now spans several segments, and one that does not fit
the current batch resumes in the next. Verified on a synthetic 3M-token shard: token use went from
80.2% to **100.0%**, with no duplicated tokens, no zero-length segments, and no segment over
`max_seq_len`. Batch count rose 24%.

**Open risk:** this leaves only about 74 steps of margin (0.24%) between what the data yields
(~30,520 steps) and what the schedule wants (30,446). If the real shards come in slightly under 100M
tokens, the run falls short again and the cosine still will not complete. One extra shard
(`NUM_TRAIN_CHUNKS=11`) would buy ~10% headroom.

---

## 2026-09-09 — Post-mortem: the qwen3_50m H100 run

Base commit: `0deefe6` on `main`. Run: `qwen3_50m` on one H100, W&B run `4pepd8cg`. It died at
step 20490 of 30446 with a NaN loss. I went looking for the NaN and found a bigger problem instead.

### The weight matrices never trained

`muon_lr` was `0.0002`. Muon orthogonalises the gradient before it applies it, so the singular
values are always about 1. That means the learning rate *is* the step size in weight space. It does
not scale with the gradient the way an AdamW learning rate does. The standard band is 0.02 to 0.05.

At 0.0002 a 512x512 matrix moved about 0.04% per step. After 20,000 steps every 2D weight was still
sitting at its random initialisation. The numbers are blunt about it:

- Every spectral metric is flat. Max divided by min over the whole run: effective_rank 1.0007,
  n_for_99pct 1.0000, stable_rank 1.0151, condition_number 1.0348.
- Power-law alpha is above 10 on 53 of 56 layers. `METRICS.md` calls that "white noise, layer is
  untrained". No layer reached the healthy 2-4 band.
- effective_rank sat at 95% of its maximum. That is maximum entropy, which means no structure was
  learned.
- Loss stalled at 5.94, so perplexity 380. That is roughly bigram quality.

Only the 33 one-dimensional norm gains moved, and those run on AdamW. The cause is simple. The H100
config raised the batch 16x, from 2048 to 32768 tokens per update, and inherited the old learning
rates unchanged. The config carried a comment saying the LRs may need scaling. Nobody acted on it.

One nice piece of confirmation. `o_proj` condition numbers looked alarming, up to 66,770 against a
model median of 3.67. But group the matrices by shape and the pattern is exact: only the square ones
(`q_proj`, `o_proj`) misbehave, and every rectangular shape sits at about 3. That is the
random-matrix signature. A square random matrix has its smallest singular value pressed near zero. A
trained one would have settled. So the scary number was more evidence that nothing trained.

### The NaN had no guard anywhere

Loss was a healthy 5.94 at step 20200 and NaN by 20250. Nothing in the trainer checks for it.
`clip_grad_norm_` does not help, because `error_if_nonfinite` defaults to False, so clipping a NaN
norm does nothing. The unconditional `optimizer.step()` then wrote NaN into every parameter. The run
kept going for another 250 steps on garbage. It only stopped because `np.histogram` in the analysis
hook hit a NaN gradient and raised.

It was not an explosion. The largest values reached anywhere in the run were activations 9.10,
weights 4.38, gradient elements 0.14, gradient norms 3.17. bf16 tops out at 3.4e38. So nothing grew.
The NaN came from an operation, not from magnitude.

### The packer can hand a zero-length segment to flash attention

`cu_seqlens` gets clamped to the input length. The buffer holds `packed_tokens+1` and the inputs hold
`packed_tokens`, so the clamp pulls the last boundary down by one. If the final segment has length 1,
it becomes length 0 and leaves a duplicated boundary. `_doc_position_ids` was already hardened for
exactly this case, but the same `cu_seqlens` went into `flash_attn_varlen_func` unfiltered, where a
zero-length segment is undefined behaviour. Simulating the packer: it happens about 39 times in a
20,490-step run. That makes it the leading suspect for the NaN, but it is not proven.

### The gradient SNR metric was lying

The EMA variance started at 1.0 instead of 0. Real gradient variance is around 1e-6, and with beta
0.99 updating only every 50 steps, that seed needs about 69,000 steps to decay. The run reached
20,490. So the reported SNR was mostly the leftover initial value.

Worse, the slowly rising SNR read as healthy convergence. It was not. The predicted rise from the
bias decaying alone is 7.10x. The observed rise was 7.62x. So the real signal grew about 7% in 20,000
steps and the rest was measurement error. On a synthetic test with a known SNR of 0.5, the old code
reported 0.0037. That is 136x off.

### Three smaller bugs found on the way

- RoPE ran before QK-norm. Qwen3 applies QK-norm first. The two orders are not the same, because the
  norm has a learnable per-channel weight and RoPE has already mixed the channels by then.
- `ignore_index` was set to `pad_token_id`. GPT-2 has no pad token, so the tokenizer aliases it to
  eos, 50256. That token is the document separator in the FineWeb data. So every end-of-document
  target was silently dropped from the loss.
- Eval never moved `cu_seqlens` to the GPU. That would have crashed at the very end of a successful
  run.

### Changes made

| File | Change |
|---|---|
| `configs/training/qwen3_50m_h100.yaml` | `muon_lr` 0.0002 to 0.02 |
| `optimizer/muon.py` | Embeddings and the tied head now go to AdamW, not Muon. Added the `max(1, rows/cols)**0.5` step scale. |
| `trainer/trainer.py` | Skip the batch when loss or gradient norm is not finite, and log enough to identify it. Abort after 20 in a row. `ignore_index` 50256 to -100. Move `cu_seqlens` to the device in eval. |
| `datasets/fineweb_helper.py` | Drop duplicate boundaries, so no zero-length segment reaches attention. |
| `layers/attention/gqa.py` | QK-norm now runs before RoPE. |
| `callbacks/analysis_callback.py` | EMA variance starts at 0 with bias correction. The histogram bins only finite values, so a NaN gradient cannot kill the run before the trainer can skip it. |

The old checkpoint is not worth resuming. Those weights learned nothing, so there is nothing to
recover. Starting fresh.

### What to check next

Run about 2000 steps and watch two numbers. Alpha should fall from above 10 toward 2-6. The condition
number on the square matrices should stop swinging 30x and start to settle. If both move, the
learning rate was the problem. If they stay frozen, the theory is wrong and it needs another look.

### Still open

- No checkpoint resume exists. `CheckpointCallback` only saves, and `get_exp_path` always makes a
  fresh timestamped directory. `resume_wandb_id` / `WANDB_RUN_ID` only reattach the W&B logging run;
  they restore no model, optimizer, or step state, which silently corrupts the old run's history.
- `_epoch_eval` materialises full logits (32768 x 50304, about 3.3 GB in bf16) instead of using the
  fused CE path the training loop uses. That is an OOM risk at end of run.
