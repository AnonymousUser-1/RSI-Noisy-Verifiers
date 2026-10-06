# Matched Error Rates Do Not Fix Learning: code and records

Anonymous code and data release for the AISTATS 2027 submission *Matched Error Rates Do Not Fix
Learning: Influence Profile and Budgeted Auditing in Iterative Self-Training*.

The release has three parts:

| Part | Where | What it lets you do |
|---|---|---|
| Implementation | `rsi/`, root `*.py`, `experiments/`, `scripts/multiround/`, `configs/`, `tests/` | Generate the tasks, draw candidate pools, build matched R/S selections, train LoRA arms, audit, evaluate |
| Run records | `experiments/outputs/` | The saved per-round records (selection, training, audit and evaluation summaries) that every reported number is computed from |
| Paper checks | `manuscript/` | The LaTeX source, the scripts that built the result tables, and scripts that re-check every table against the run records |

Model weights, LoRA adapters, raw sampled answers and most raw evaluated answers are not included
(section 5).

## 1. Install

Python 3.10-3.12 is recommended.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-core.txt                    # CPU checks: numpy, scipy, matplotlib
pip install -r requirements-gpu.txt                     # GPU pipeline: torch, transformers 4.57.1, peft 0.17.1
pip install pytest reportlab pypdf                      # tests and the table/figure scripts
```

Keep files byte-exact: the checks compare SHA-256 digests. `.gitattributes` disables line-ending
conversion in git; if you copy files by other means, do not let a tool rewrite line endings.

## 2. What a reviewer can check on a CPU (minutes)

Commands run from the repository root unless stated.

**Unit tests** (about one minute; the shell-script tests need `bash`):

```bash
python -m pytest tests -q
```

**Theory.** Numerical checks of the logistic model, the population flow and the audit threshold
(Section 4 and Appendix A):

```bash
python theory_checks.py --code-dir .
```

**Task data.** The graph and arithmetic splits are generated deterministically (seed 2027);
`scripts/multiround/01_make_data.sh` checks the result against fixed fingerprints:

```bash
python generate_data.py --task graph --seed 2027 --out data/graph
python generate_data.py --task arithmetic --seed 2027 --out data/arithmetic
```

**Paper tables against the run records.** From `manuscript/`:

```bash
cd manuscript
python theory_checks.py --code-dir ..
python verify_h100_tables.py                      # Llama-3.2-3B four-round tables
python verify_graph_extensions.py --code ..       # Qwen3-1.7B four-round and all one-step tables
python verify_qwen_raw_figures.py --code ..
python verify_added_raw_figures.py --code ..
python verify_qwen4b_raw_figures.py --code ..
```

Each reports its verification scope. Checks that also use the original Qwen3-4B task files or
candidate answers take an optional archive path; see [original-input checks](OPTIONAL_INPUTS.md).
The ordinary experiment pipeline does not require that archive.

The summaries behind the tables can be regenerated from the records into a new directory:

```bash
python summarize_h100_results.py --snapshot ../experiments/outputs/2026-10-03-h100 --out /tmp/llama
python summarize_h100_tpr.py --snapshot ../experiments/outputs/2026-10-03-h100 --out /tmp/tpr --table /tmp/tpr.tex
python summarize_graph_extensions.py --code .. --out /tmp/graph_extensions
```

**Audit-threshold figures** (Figure 2):
`python manuscript/RSI-audit-figures-20261003/reproduce.py --out-dir /tmp/audit` regenerates the scan
data byte-for-byte and redraws the figures.

**Paper.** `manuscript/main.tex` compiles with Tectonic or XeLaTeX + BibTeX.

## 3. Map from the paper to the records

| Paper | Records | Table scripts |
|---|---|---|
| Llama-3.2-3B four-round graph, audited and unaudited | `experiments/outputs/2026-10-03-h100/graph_{un,}audited_llama3.2-3b/` | `summarize_h100_results.py`, `summarize_h100_tpr.py`, `verify_h100_tables.py` |
| Qwen3-1.7B four-round graph, audited and unaudited | `experiments/outputs/2026-10-03-h100/graph_{un,}audited/` | `summarize_graph_extensions.py`, `verify_graph_extensions.py` |
| Qwen3-4B four-round graph, unaudited | `experiments/outputs/2026-10-05-qwen3-4b-multiround-unaudited/` | `summarize_qwen4b_results.py`, `summarize_qwen4b_tables.py`, `verify_qwen4b_results.py` |
| Qwen3-4B four-round graph, audited (B = 16) | `experiments/outputs/2026-10-05-qwen3-4b-graph-audit-b16/` | same |
| One-step R/S/null, three models | `experiments/outputs/2026-10-04-one-step/` | `summarize_graph_extensions.py`, `verify_added_raw_figures.py` |
| Arithmetic (partial; training diagnostics only) | `experiments/outputs/2026-10-03-h100/arithmetic_*` | `summarize_h100_results.py` |
| Optional Qwen3-4B original inputs | Separate archive, described in [OPTIONAL_INPUTS.md](OPTIONAL_INPUTS.md) | `verify_inputs.py` in the extracted archive |

Notes:

- Several display tables (`manuscript/h100_*_table.tex`, `qwen4b_multiround_endpoint_table.tex`) were
  typeset from the generated summaries; the `verify_*` scripts check every displayed cell against the
  records.
- The graph runs in `2026-10-03-h100/` were later extended from four to eight rounds; the paper reports
  rounds 1-4, and the `*_rounds0-4_*` summaries hold exactly those. `manuscript/figures/h100_runs/provenance.json`
  records the digests of the earlier four-round snapshot, so its entries for the files rewritten by the
  extension (`experiment.json`, the first `run.json` per arm, the un-suffixed summaries) no longer match.
- The training-diagnostics table of the initial, partially completed series was built from that
  earlier snapshot. `summarize_h100_results.py` regenerates it from the complete records, so its
  `partial_training.csv` has more blocks per row than the table.
- The run records were written on the original machines. Absolute paths and host names inside them
  were replaced by neutral placeholders (`/cluster`, `/cluster2`, `/local`, `C:\Users\user`,
  `workstation`), some folder names were shortened, snapshot times are given in UTC, and internal
  review and job identifiers were replaced by neutral labels; nothing else in them differs from what
  the runs wrote. Every digest that the records and the verification scripts carry was recomputed after
  the replacement.

## 4. Running the GPU pipeline

The full pipeline needs a Linux machine with an NVIDIA GPU (H100/A100 class for the four-round runs)
and the pinned Hugging Face models (`matched-dynamics/base_pin*.json`):

| Model | Commit |
| --- | --- |
| `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |
| `meta-llama/Llama-3.2-1B-Instruct` | `9213176726f574b556790deb65791e0c5aa438b6` |
| `meta-llama/Llama-3.2-3B-Instruct` | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |

The Llama repositories are gated on the Hub. Qwen3 runs with `enable_thinking=False`. The Llama 3.x
chat template's date is fixed to `26 Jul 2024` (`rsi.backends.CHAT_TEMPLATE_KWARGS`).

```bash
bash scripts/multiround/00_setup.sh ~/venvs/rsi
export PY=~/venvs/rsi/bin/python RSI_ROOT=~/rsi-work
bash experiments/run.sh graph_unaudited                 # one four-round experiment, end to end
bash experiments/one_step.sh graph_unaudited            # the one-step R/S/null comparison on its round-1 pools
```

`experiments/README.md` lists the experiments, their settings and run times;
`MULTIROUND_EXPERIMENT_INSTRUCTION.md` and `ONE_STEP_EXPERIMENT_INSTRUCTION.md` give the full protocols.
Qwen3-4B uses the same stage driver. In a fresh `RSI_ROOT`, generate its inputs and run the
configured multi-round experiment, then the paired one-step experiment:

```bash
bash experiments/run.sh graph_unaudited_qwen3-4b
bash experiments/one_step.sh graph_unaudited_qwen3-4b
```

No original-input archive is required. The one-step run uses the pools, initialization and
first-round matching produced by this new multi-round run. Original-input replay is a
separate, optional operation in [OPTIONAL_INPUTS.md](OPTIONAL_INPUTS.md).

Notes on reproducing specific runs:

- The graph experiments' `config.json` set `"rounds": 8`; rounds 1-4 of an eight-round run are those
  of a four-round run.
- The Qwen3-4B four-round runs used a run-specific controller that switched the repetition stop on
  part-way (block b01 round 4 onward) and trained R and S in parallel. Every unit's setting is recorded
  (`protocol_profile.json`, `protocol-receipt.json`). `experiments/run.sh` applies one setting to every
  round. The fresh Qwen3-4B configuration retains `repetition_stop: null` throughout sampling;
  it does not recreate the historical per-round switch. Reproducing that mixed schedule is
  outside this entry-point revision. The archived result records remain the evidence for the
  reported four-round results. This release has no audited Qwen3-4B configuration folder.
- Greedy evaluation is not bit-identical across GPU types; the paper reports each run's own baseline.
- Some modules (`rsi/experiment.py`, `rsi/paired_evaluation.py`, `rsi/reference_gradient.py`,
  `evaluate_one_step.py`, `run_experiment.py`, and the queue in `launch.py`/`prepare_suite.py`) belong to
  an earlier design and did not produce the reported results; they remain because the tests cover them.

## 5. What is not included, and what that limits

Not included:

- Base model weights (download them by the commits above).
- The LoRA adapters of every trained round and one-step arm, the reference-gradient tensors `h.pt` of
  the H100 and one-step runs, later-round candidate pools, audit ledgers, and the raw evaluated answers of
  the H100 and one-step runs.
- Compute-cluster logs and job scripts.

What each diagnostic can therefore be checked against:

| Diagnostic | Check available in this release |
|---|---|
| Matching certificates (N+, N-, TPR, FPR, yield, target hits), selection counts | Recomputed from the saved selections and counts (`verify_*.py`) |
| Round-1 judging of the Qwen3-4B candidates | With the optional archive: 81,920 answers, `summarize_qwen4b_tables.py --check --inputs INPUTS` |
| Audit labels, retained counts, query totals of the Qwen3-4B audited run | With the optional task files: 123,000 answers, `summarize_qwen4b_results.py --inputs INPUTS`; the verifier checks saved results and hashes |
| Greedy Pass@1, target-error rates, paired contrasts and intervals | Recomputed from the saved per-checkpoint counts; the per-question answers are omitted, so they are not rejudged |
| `h^T Delta theta`, `||Delta theta||` | Checked as recorded summary values only. The adapters are omitted. `scripts/reproject.py` recomputes both from adapters on a CPU; on the authors' copies of the Qwen3-4B adapters it reproduces the records (for example b00/R round 1: -0.019447912664691 recorded and recomputed). The Qwen3-4B reference gradient is in the optional archive (`reference_out_audit_*/h.pt`, the `h_sha256` of both Qwen3-4B four-round runs) |
| Reference losses and per-sample reference terms | Recorded summary values only |

For the optional projection check, set `INPUTS` to the extracted archive and supply the
trained adapters separately (they are not included):

```bash
python scripts/reproject.py --h "$INPUTS/reference_out_audit_new/h.pt" \
  --start "$INPUTS/shared_adapter" \
  ADAPTERS/b00/R/round_001/adapter ADAPTERS/b00/R/round_002/adapter \
  --records experiments/outputs/2026-10-05-qwen3-4b-graph-audit-b16/out/b00/R/round_001 \
            experiments/outputs/2026-10-05-qwen3-4b-graph-audit-b16/out/b00/R/round_002
```

## 6. Files and licenses

`MANIFEST.md` lists every file of the release with its size and SHA-256 digest. The code is released
under the MIT License (`LICENSE`); models, libraries and external datasets keep their own licenses
(`THIRD_PARTY_NOTICES.md`).

## 7. Layout

```text
rsi/                      library: tasks and judges, matching, selection, auditing, LoRA backend, theory model
*.py                      entry points (sampling, matching, training, evaluation, reference gradient)
experiments/              one folder per experiment (config.json, evaluation.json, settings.sh) and drivers
experiments/outputs/      run records used by the paper
scripts/multiround/       stage scripts used by experiments/run.sh; scripts/reproject.py recomputes h^T Delta theta
configs/                  matched-line configurations (configs/README.md lists every field)
matched-dynamics/         protocol notes and the model pins
manuscript/               paper source, table scripts and verification scripts
OPTIONAL_INPUTS.md        optional original-input archive and stronger data-level checks
tests/                    unit tests
```
