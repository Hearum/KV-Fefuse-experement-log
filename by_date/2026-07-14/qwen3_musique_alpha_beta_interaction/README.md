# Qwen3 MuSiQue Alpha-Beta Interaction Probe

## Background

This experiment continues the MuSiQue-only attention alpha and KV beta probes.

Existing facts from earlier notes:

- `alpha` changes only selected-token recompute attention output:
  `mixed_attn_out = (1 - alpha) * real_attn_out + alpha * ablated_attn_out`.
- `uniform/random alpha=0.1~0.25` improved MuSiQue rate=0.15 DraftModel accuracy, while large alpha destroyed performance.
- The early beta sweep was not a clean beta test because selected document tokens and query tokens were prefetched in one forward; beta blend happened after the query had already seen full recompute KV.
- The corrected beta interface is `FUSIONRAG_STRICT_BETA_QUERY_PREFILL=1`: recompute selected document tokens first, apply beta blend, then run query prefill as a separate forward.

## Question

On MuSiQue, is the alpha gain mainly:

1. a recompute-attention smoothing effect;
2. a damped KV-writeback effect;
3. or an interaction between the two?

## Clean Semantics

Selection remains native online DraftModel with `rate=0.15`.

For selected document tokens:

1. recompute their KV, optionally with alpha attention-output perturbation;
2. apply strict beta blend before query prefill:
   `KV_after = beta * KV_cached + (1 - beta) * KV_recomputed`;
3. run query prefill after the document KV has already been blended.

So:

- `alpha=none, beta=0` is the strict version of native selected-token full KV update.
- `alpha=none, beta=1` is the strict no selected-doc-KV update control, close to true `rate=0` in mechanism.
- `alpha>0, beta=0` asks whether alpha alone helps.
- `alpha>0, beta=0.5` asks whether alpha remains useful after damping the final KV delta.

## Phase 1: MuSiQue 0-50 Sanity Grid

Use 50 samples first. MuSiQue has 200 sub-question rows; previous full alpha tables aggregate these into 135 main questions / 248 sub-questions, so full expansion should use `0:200`.

Sanity grid:

| label | alpha mode | alpha | beta | purpose |
|---|---|---:|---:|---|
| native_beta0 | none | 0 | 0 | strict full selected-token KV update |
| native_beta0p5 | none | 0 | 0.5 | pure beta damping |
| native_beta1 | none | 0 | 1 | strict no selected-doc-KV update control |
| uniform_a0p1_b0 | uniform | 0.1 | 0 | alpha effect only |
| uniform_a0p1_b0p5 | uniform | 0.1 | 0.5 | alpha plus damped KV |
| uniform_a0p25_b0 | uniform | 0.25 | 0 | stronger alpha effect only |
| uniform_a0p25_b0p5 | uniform | 0.25 | 0.5 | stronger alpha plus damped KV |
| random_a0p1_b0 | random | 0.1 | 0 | noise alpha effect only |
| random_a0p1_b0p5 | random | 0.1 | 0.5 | noise alpha plus damped KV |

Decision rule before expansion:

- If beta=0.5 beats beta=0 without alpha, beta damping is itself useful.
- If alpha beta=0 beats native beta=0, alpha is independently useful under strict semantics.
- If alpha beta=0.5 beats both alpha beta=0 and native beta=0.5, the useful effect is interaction.
- If beta=1 is close to beta=0, selected doc-KV update may be less important than expected.

## Paths

Output root:

`/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_alpha_beta_interaction_20260714`

Cache root:

`/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache`

The cache should be reused read-only; this experiment must not create a per-worker duplicate cache pool.


## 2026-07-14 Launch: Phase 1 Sanity

Commit before launch: `1ec39dd exp: add musique alpha beta interaction probe`.

Launched on qjy000 using GPUs 4,5,6 only, leaving qjy000 GPUs 0-3 for the existing cross-dataset alpha rate sweep.

Command:

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_alpha_beta_interaction_20260714
FUSIONRAG_MUSIQUE_ALPHA_BETA_GPUS="4,5,6" \
FUSIONRAG_MUSIQUE_ALPHA_BETA_START=0 \
FUSIONRAG_MUSIQUE_ALPHA_BETA_END=50 \
FUSIONRAG_MUSIQUE_ALPHA_BETA_CACHE_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache \
nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/qwen3_musique_alpha_beta_interaction/scripts/run_musique_alpha_beta_sanity.py \
  > "$ROOT"/logs/launcher_qjy000_0_50_gpu456.nohup.log 2>&1 &
```

Launcher PID: `478310`.

Important environment semantics:

- `FUSIONRAG_STRICT_BETA_QUERY_PREFILL=1` is set by the launcher for every child job.
- `FUSIONRAG_PREPROCESS_CACHE_READONLY=1` is set by the launcher; cache root is the canonical shared cache.
- `FUSIONRAG_REPROCESS_ATTENTION_CHUNK=64` is set to avoid large attention-ablation tensors.

Summary command:

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
FUSIONRAG_MUSIQUE_ALPHA_BETA_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_alpha_beta_interaction_20260714 \
  MOTIVATION_EXPERIMENTS/qwen3_musique_alpha_beta_interaction/scripts/summarize_musique_alpha_beta.py
```

## 2026-07-14 Initial Runtime Check

Runtime check shortly after launch:

- Active qjy000 processes: launcher plus three workers on GPUs 4,5,6.
- First running tasks: `native_beta0`, `native_beta0p5`, `native_beta1` on MuSiQue `0:50`.
- Three partial CSVs are already being written, but none has `FINAL RESULTS` yet.
- Current partial rows are only around 5 rows per task, so they are not used as evidence and no conclusion should be drawn yet.

Next action after the 9-task sanity grid finishes:

1. run `scripts/summarize_musique_alpha_beta.py`;
2. append the completed table here;
3. decide whether full MuSiQue `0:200` should expand the best alpha-beta combinations or only the beta-only strict grid.
