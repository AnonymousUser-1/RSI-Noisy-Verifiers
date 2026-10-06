# C: independent one-step evaluation

This is a separate entry point; `evaluate.py` and `analyze.py` continue to support
legacy round-based experiments. Supported truth functions are exactly the
`rsi.tasks.judge` checkers: graph, arithmetic, and the imported GSM8K and DeepMind
Mathematics tasks (README.md). No LLM judge is used. A dataset without eval_ood (GSM8K)
is frozen and evaluated on eval_id alone.

## Before formal training

Freeze a **fully resolved** model config, dataset and evaluation source version:

```bash
python evaluate_one_step.py freeze --config resolved_config.json --data data/main/graph --out evaluation_protocol.json --draws 4 --seed 2027 --null-repeats 1 --bootstrap 2000 --bootstrap-seed 42
```

The numerical choices above are examples, not an automatic protocol amendment.
Choose them using development cost measurements before confirmatory training.
The config must explicitly contain backend, model, immutable revision (HF commit),
device, dtype, and the generation batch size, sequence limit, completion limit,
temperature, top_p and top_k. No training defaults are inferred or changed.
All dataset manifest splits are checked for overlapping instance IDs.
Source changes after freezing require an explicit new freeze and protocol decision;
this entry point refuses silently evaluating with changed code.

## B to C handoff (schema version used by this entry point)

B's current `one_step.json` lacks complete provenance. Supply an additional JSON
handoff; paths are relative to the handoff file, or absolute. Include **all** formal
blocks together, each with a distinct ID and training seed. Example block:

```json
{
  "blocks": [{
    "id": "seed0",
    "seed": 0,
    "base_model": "Qwen/Qwen3-1.7B",
    "base_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "adapters": {
      "initial": "shared_adapter",
      "R": "seed0/R/adapter",
      "S": "seed0/S/adapter"
    },
    "adapter_hashes": {"initial": "REPLACE", "R": "REPLACE", "S": "REPLACE"},
    "pool": "pools/seed0.jsonl",
    "pool_hash": "REPLACE",
    "matched": "matched_subsets.json",
    "matched_hash": "REPLACE",
    "one_step": "seed0/one_step.json",
    "one_step_hash": "REPLACE",
    "training": {
      "R": {"completed": true, "optimizer_steps": 1, "initial_parameter_hash": "REPLACE"},
      "S": {"completed": true, "optimizer_steps": 1, "initial_parameter_hash": "REPLACE"}
    }
  }]
}
```

Use `rsi.paired_evaluation.tree_hash(directory)` for adapter directory hashes and
`rsi.common.file_hash(path)` for artifact hashes. `initial_parameter_hash` is B's
recorded actual starting tensor hash, matching `shared_adapter.json`; it is NOT the
directory hash. B must measure and record this before the update. Do not invent a
measurement by copying an expected value after training. Initial adapter metadata
must contain `base_model`, `base_revision`, `parameter_hash`.

This handoff is an interface requirement, **not a repair of B's training code**.
Completion and initial tensor hashes are producer attestations. C verifies
consistency and file integrity, but cannot prove optimizer execution history from
an adapter alone. Known accumulation/diagnostic defects must be fixed by B before
formal use. K must be 16, 32 or 64 and identical across all supplied blocks; C
rechecks actual subsets but does not independently prove largest-K optimality,
random selection uniformity, or that the supplied block list is the registered
complete list. Those remain checks against the frozen study manifest/producer log.

## Run and analyze

```bash
python evaluate_one_step.py run --protocol evaluation_protocol.json --handoff handoff.json --out evaluation/run1
python evaluate_one_step.py analyze --evaluation evaluation/run1 --out analysis/run1
```

All blocks are validated before final-test generation. C independently rejudges
original candidates, checks unique prompt/candidate membership, common correct
IDs, 3:1 counts, uniform weights and exact per-prompt supervised token counts.
Lengths are re-tokenized as response + one real EOS, matching the current training
encoding, not copied from `completion_tokens`. The same original pool supplies
TPR/FPR denominators. The new joint procedure has no legacy provisional stage;
missing stages are explicitly identified, not fabricated.

The initial arm always loads the shared adapter. R, S and initial use the same
per-block/split sampling seed and batching. Null repeats reload the shared adapter
and use distinct repeat seeds. Repeat numbers and all decoding parameters are frozen.
A shared seed does not guarantee identical random trajectories across models.

Evaluation requires fresh output directories. Interrupted evaluations have no
`complete.json` and are rejected by analysis; rerun into a new directory. This
version intentionally does not resume partial inference. It writes per-answer
JSONL, per-arm wall times, a provenance manifest, matching audit and output hashes.
`reached_length_limit` is a length diagnostic, **not** an EOS/finish-reason claim:
the existing generator does not expose the actual stop reason.

Analysis writes:
- `metrics_per_block.csv`: accuracy/error and error-type counts;
- `paired_contrasts.csv`: R-initial, S-initial, S-R and null-repeat-initial;
- `block_summary.csv`: mean paired contrasts and training-block t intervals;
- `analysis_manifest.json`: provenance, interpretation and demo status.

Each prompt's answer errors are averaged before averaging prompts (sampled pass@1,
never pass@k). Prompt bootstrap samples whole paired prompts, retaining all draws.
It conditions on these trained models and does not estimate training-seed variation.
Block t intervals use complete paired blocks, not prompts or answer draws, as the
independent units. Three blocks give very uncertain intervals; no regression or
universal safety claim is supported. No multiplicity correction is automatic;
predeclare the primary ID S-R contrast, treat other contrasts as secondary.
Positive S-R means S is worse than R, not necessarily worse than initial.

Mock mode is for plumbing only: synthetic word-based lengths, fake model outcomes,
`DEMO_ONLY` in both manifests, excluded from analysis unless `--include-demo` is
explicit. No GPU or real-model result is claimed by the CPU tests.

## GPU smoke test

From the repository root, on a GPU node with `requirements-gpu.txt` and `pytest` installed:

```bash
python evaluation_gpu_smoke.py slurm-local/results/manual-smoke
```

The output directory must be new. This runs the whole test suite under pytest in
its own process, without failures or skips (report in `tests.xml`),
checks weighted loss gradients on CUDA, saves and reloads a LoRA adapter, checks
that the reloaded model reproduces the forward pass bit for bit, and samples
graph and arithmetic examples with a fixed seed before and after the reload.
Whether those sampled texts match is recorded, not asserted: on an H100,
Llama-3.2-1B flips near-tied tokens between two fixed-seed runs of one model
instance, so sampled decoding is not bitwise reproducible. It uses the study's
base pin: the pinned Qwen3-1.7B revision above, or the pin file `RSI_BASE_PIN`
names. `--model ID --revision COMMIT` checks another checkpoint instead, such as
the Llama-3.2 models pinned in `README.md`. Cache the
model first if running offline; set
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to prevent downloads.
This is a backend plumbing check, not a complete paired-evaluation run or a
scientific training experiment. GPU execution and driver compatibility must be
verified on the target cluster. Machine-specific Slurm launchers and all local
artifacts remain ignored under `slurm-local/`.
