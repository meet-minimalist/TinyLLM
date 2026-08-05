# Downstream Evaluation Benchmarks

TinyLLM runs five zero-shot downstream benchmarks *during* training to track model
quality beyond the training loss. They are **log-likelihood** benchmarks — the model
is never asked to generate text. Instead, for each example we score the candidate
answers by how much probability the model assigns them and pick the highest. This
makes them cheap, deterministic, and meaningful even for tiny, partially-trained
models.

They are scheduled by `BenchmarkCallback` (`callbacks/benchmark_callback.py`), executed
by `run_benchmarks` (`benchmarks/runner.py`), and logged to WandB as
`benchmark/<name>/<metric>`.

---

## The benchmarks at a glance

| Benchmark  | Task | HF dataset (split) | Full split size | Samples we use | Answer options | Random baseline | Metric(s) |
|------------|------|--------------------|-----------------|----------------|----------------|-----------------|-----------|
| **LAMBADA**    | Predict the final word of a passage | `EleutherAI/lambada_openai` (test) | ~5,153 | **500** | none — open vocabulary | ~0% | accuracy + perplexity |
| **HellaSwag**  | Pick the most plausible sentence ending | `Rowan/hellaswag` (validation) | ~10,042 | **200** | **4 options** (4-way) | 25% | accuracy |
| **WinoGrande** | Resolve an ambiguous pronoun | `allenai/winogrande` `winogrande_xl` (validation) | ~1,267 | **300** | **2 options** (2-way) | 50% | accuracy |
| **ARC-Easy**   | Elementary-science multiple choice | `allenai/ai2_arc` `ARC-Easy` (validation) | ~570 | **200** | **4 options** (4-way)¹ | 25% | accuracy |
| **PIQA**       | Physical-commonsense QA | `lighteval/piqa` (validation) | ~1,838 | **300** | **2 options** (2-way) | 50% | accuracy |

Grouped by answer format:
- **2-option (binary choice):** WinoGrande, PIQA → random = 50%.
- **4-option (multiple choice):** HellaSwag, ARC-Easy → random = 25%.
- **Open-vocabulary (no options, exact-match):** LAMBADA → random ≈ 0%.

¹ ARC-Easy questions are *mostly* 4-way but a small number have 3 or 5 choices; the code
scores whatever number of choices each question actually provides, so 25% is the nominal
(not exact) random baseline.

`Samples we use` is the `num_samples` set per benchmark in `train_config.yaml`. We
evaluate on a **random but fixed subset** (shuffled with `seed=42`) rather than the full
split, so every evaluation during a run scores the *same* examples — curves are
comparable across steps and cheap enough to run mid-training. Raise `num_samples`
(toward the full split size) for lower-variance final numbers.

---

## How the model is judged

### Log-likelihood scoring (`base.py::score_completions`)

For a shared `context` and a list of candidate `completions`, we:

1. Encode `context + completion` as one sequence.
2. Take the model logits over **only the completion tokens** (everything after the
   context length).
3. Compute the mean per-token cross-entropy of those completion tokens, and set
   `score = -mean_cross_entropy` (i.e. the **mean log-probability per completion token**).
4. The completion with the **highest score wins**. It's counted correct if that index
   equals the gold `label`.

Because the score is a **per-token mean** (length-normalized), a longer answer is not
unfairly penalized for having more tokens. HellaSwag, WinoGrande, ARC-Easy, and PIQA all
use this. Only the differing part of each option is scored (e.g. WinoGrande splits the
sentence at the blank and scores `option + suffix` against the shared prefix), which
isolates the real signal and further reduces length bias.

### LAMBADA is different (exact-match, not multiple choice)

LAMBADA has no answer choices — the model must produce the exact final word. We:

- Feed the passage up to the last word, then **greedily `argmax`** the model's
  predictions at the final-word token positions.
- Count the example correct only if **every token** of the last word matches exactly.
- Also accumulate cross-entropy on those tokens and report **perplexity =
  `exp(mean_loss)`**.

So LAMBADA reports two numbers: `accuracy` (exact last-word recovery) and `perplexity`
(lower is better).

---

## How to read the numbers

**1. Compare to the random baseline, not to zero.** A 4-way benchmark at 25% means the
model has learned nothing discriminative yet; 2-way at 50% is chance. Early in training
you should expect values *near baseline* — e.g. at step 100 you'll see HellaSwag ≈ 0.28,
WinoGrande ≈ 0.53, ARC-Easy ≈ 0.29. That is normal, not a bug.

**2. Expected trajectory by scale** (rough guide from the benchmark docstrings):

| Benchmark  | ~50M model | Large (≈3B / GPT-3) | Emerges… |
|------------|-----------|----------------------|----------|
| LAMBADA    | ≈25%      | ≈76% (GPT-3 175B)    | early, climbs smoothly |
| PIQA       | ≈60%      | ≈77%                 | **earliest** signal above chance |
| WinoGrande | ≈51%      | ≈64%                 | late (needs coreference ability) |
| ARC-Easy   | ≈33%      | ≈65%                 | late (needs factual knowledge) |
| HellaSwag  | ≈28–30%   | high                 | mid/late (needs scale) |

**3. Which benchmarks to watch when.** PIQA and LAMBADA move first, so they're the best
early-training health signals. ARC-Easy and WinoGrande stay near random for small models
and only lift with scale/knowledge — don't be alarmed if they're flat early.

**4. Account for sampling noise.** With a subset of `n` examples, the 95% confidence
interval on an accuracy `p` is roughly `±1.96 · sqrt(p(1-p)/n)`. For `n=200` at `p≈0.5`
that's **±7%**; for `n=500` it's **±4.4%**. Treat single-eval wiggles inside that band as
noise — trust the *trend* across many evaluations, not one point. Increase `num_samples`
to tighten the interval.

**5. Judge quality holistically.** No single benchmark is decisive at this scale. Look
for: LAMBADA perplexity trending down, PIQA rising above 50%, and the 4-way tasks
eventually pulling above 25%. Training loss going down while benchmarks stay flat usually
just means the model is too small / undertrained for those tasks yet.

---

## Configuration (`train_config.yaml` → `benchmarks:`)

```yaml
benchmarks:
  lambada:
    enabled: true
    num_samples: 500       # of ~5K test examples
    every_n_steps: 5000    # 1 forward/example → cheap, run often
  hellaswag:
    enabled: true
    num_samples: 200       # 4-way → 4 forwards/example
    every_n_steps: 100
  winogrande:
    enabled: true
    num_samples: 300       # 2-way → 2 forwards/example
    every_n_steps: 100
  arc_easy:
    enabled: true
    num_samples: 200       # 4-way → 4 forwards/example
    every_n_steps: 100
  piqa:
    enabled: true
    num_samples: 300       # 2-way → 2 forwards/example
    every_n_steps: 100
```

- **Per-benchmark scheduling.** Each benchmark has its own `every_n_steps`; only those
  *due* at the current step run (`BenchmarkCallback`). Intervals are staggered — LAMBADA
  runs less often because it's cheap but we still want a smooth curve, while the
  multiple-choice tasks (more forwards/example) share a common cadence.
- **Cost intuition.** Runtime ≈ `num_samples × choices` forward passes. LAMBADA is the
  cheapest (1 forward/example); the 4-way tasks are ~4× per example.
- **Disable one** by setting its `enabled: false` (or removing its block).
- **Resilience.** If a benchmark fails (dataset outage, HF API change), it's logged and
  **skipped** — training is never aborted by an eval (`runner.py`).

---

## Where the results go

- **Logs:** `Benchmark <name>: accuracy=…, time=…s` per benchmark, plus a per-step
  `Benchmarks at step N: total_time=…`.
- **WandB:** `benchmark/<name>/accuracy`, `benchmark/<name>/num_samples`,
  `benchmark/lambada/perplexity`, and `time/benchmarks`.

## Adding a new benchmark

1. Subclass `BaseBenchmark` (`base.py`) and implement `run(model, tokenizer, device)`
   returning a dict with at least `{"accuracy": float}`. Reuse `score_completions` for
   multiple-choice tasks.
2. Register it in `_BENCHMARK_MAP` in `runner.py`.
3. Add a config block under `benchmarks:` with `enabled`, `num_samples`, `every_n_steps`.
