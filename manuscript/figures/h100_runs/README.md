# H100 snapshot results used in the manuscript

Source: `code/experiments/outputs/2026-10-03-h100`, snapshot timestamp **2026-10-03 16:06 UTC** (not live cluster status).

Only the unaudited and audited Llama-3.2-3B graph pair has all five blocks, four rounds, and both evaluation splits complete. Other runs are reported only as incomplete coverage and provisional training diagnostics. Arithmetic disables exact-length matching. The separate one-optimizer-step experiment is unmeasured in this export.

## Reproduce the numerical readouts and assets

From the manuscript directory, with Python and ReportLab installed:

```sh
python summarize_h100_results.py \
  --snapshot ../experiments/outputs/2026-10-03-h100 \
  --out /tmp/h100-results-reproduction
```

Use a fresh output directory. DejaVu Sans is preferred to match Figure 1's embedded font, with Arial as a fallback. `code/requirements-figures.txt` already includes ReportLab. No GPU, model download, or training occurs.

- `graph_accuracy.pdf`: compact main-text vector replot, same validated measurements as supplied plots; teal R / plum S and charcoal axes match Figure 1. Solid/dashed lines identify R/S; filled/open circles identify unaudited/audited runs. Its manuscript width is 70% of the text width. No point or interval is shifted or recalculated by the style change.
- `graph_target_errors.pdf`: additional compact vector target-error replot.
- `original_eval_*_pass1.png` and `original_eval_*_by_difficulty.png`: exact copies of the supplied comparison figures, included in the appendix.
- `validated_results.json`: checkpoint metrics, paired audit contrasts, matching certificates, training/composition records, coverage, feasibility, and partial diagnostics.
- `round_summary.csv`: full-precision means and descriptive intervals; `snapshot_progress.csv` and `partial_training.csv`: coverage and provisional diagnostics.
- `provenance.json`: SHA-256 bindings of 395 consumed input files, analysis script, fonts, and asset exports.

Uncertainty is pointwise 95% Student-t over **five paired blocks**, conditional on the fixed evaluation prompts; df=4. The script checks the existing plotter's critical value/intervals and computes same-arm audited-minus-unaudited error intervals from saved checkpoint summaries. It does not mistake the original plots' S-minus-R contrast for an audit effect. Round zero is a single shared base checkpoint, without five independent base fits. No multiple-testing adjustment or confirmatory inference is claimed.

The compact export omits raw candidate/evaluation answers, training rows and weight vectors, adapter weights, and `h.pt`. Validation therefore does not rejudge answers, reconstruct ESS from raw weights, reproduce GPU training, or create paired-question bootstrap intervals. Full cost accounting remains missing. The recorded 16.24 H100 GPU-hours cover only three distinct reported jobs, not the full study.

The source tables are in the manuscript's `h100_*_table.tex` files; claims and qualifications are in `results.tex` and `h100_results_appendix.tex`. No synthetic or imputed empirical outcomes are used.

Run `python verify_h100_tables.py` from the manuscript directory to compare every numeric display cell in these tables with the validated full-precision artifact. `readout_reference_checks.py` separately checks active labels, float callouts, and the five deterministic population readouts.
