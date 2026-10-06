# Selected original one-step figures

Only two raw plot types are included for each of the three completed graph
models (Qwen3-1.7B, Qwen3-4B, Llama-3.2-3B): primary ID error change against
the run's own null and actual-update reference projection. Six original
vector PDFs have model-specific readable titles; their six JSON summaries
remain byte-identical to `code/experiments/outputs/2026-10-04-one-step`.
The title-only revision verifies pixel equality below the title strip.
They appear as three paired
figures (one model per figure) in `one_step_selected_raw_figures.tex`.

This selection is based on scientific role, not favorable outcomes: every
model receives the same primary-outcome and mechanism panels. It omits OOD
error, aggregate accuracy/target-error panels, difficulty plots, displacement,
and gradient-distribution boxplots. OOD outcomes and scalar mechanism metrics
remain in the existing paper tables and three-model contrast figure.

Question: block-level heterogeneity of the primary held-out contrast and the
actual update's first-order reference diagnostic. Family: paired dot plots
and dot-and-interval contrasts, not time-series trends. Grain: five paired
blocks, one full-batch AdamW step, no correctness audit. Null is the run's own
unupdated adapter, not an independent replicate. Footprint: two original
vector PDFs at 0.82 appendix width per model, error above projection.

Original blue/red colors and axis limits are retained. Paired connectors,
R/S labels, named block points, mean diamonds, and zero lines provide non-color
distinction. Error contrasts have the opposite sign to accuracy contrasts;
negative projection is not a certified finite-step loss reduction or task
risk change. No cross-platform/model-size ranking is claimed.

`provenance.json` records selection, exact paths, source and delivered hashes,
title revisions, and the plotting script hash.
`verify_added_raw_figures.py` independently checks every plotted
block value, paired difference, and mean/SD/interval field against saved
counts/scalar diagnostics. This does not rejudge omitted answers or reconstruct
adapter/reference tensors. Final QA is the rendered manuscript appendix.
