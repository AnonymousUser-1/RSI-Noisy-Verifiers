# Original Llama-3.2-3B four-round figures

Twelve unchanged `rounds0-4` PNGs and six plotting summaries from
`code/experiments/outputs/2026-10-03-h100` are embedded as six
ID/OOD pairs in `llama_four_round_raw_figures.tex`. The grouping matches
Qwen: unaudited aggregate, unaudited difficulty, audited aggregate, audited
difficulty, joint aggregate, joint difficulty. The four old joint images
are byte-identical and now grouped consistently, not duplicated in the PDF.
Original files in `figures/h100_runs` are preserved for provenance.

Question: accuracy, paired S-minus-R accuracy, and nonshortest incidence over
the fixed four-round horizon. Family: discrete checkpoint comparisons with
intervals and task-stratum small multiples. Five checkpoints (0--4) are the
entire requested horizon; no later extension or best checkpoint is selected.
Fixed questions: 2,000 ID and 1,000 OOD, five paired blocks; common base has
no independent block replication. Palette: original blue R/red S, black
paired differences. Joint plots also use solid/circle versus dashed/square
marks, labels, and zero lines. No redraw, recoloring, cropping, or numerical
change. Footprint: two original PNGs at full appendix width per figure.

Supported reading: both arms improve; paired endpoint accuracy intervals and
same-arm audit-effect intervals include zero. Original vertical scales are
separate and intervals are descriptive, pointwise, and not adjusted for
multiple comparisons. Source hashes and sizes are in `provenance.json`.
`verify_added_raw_figures.py` checks 600 mean/SD/interval entries against the
saved four-round records. Final QA is the rendered manuscript appendix.
