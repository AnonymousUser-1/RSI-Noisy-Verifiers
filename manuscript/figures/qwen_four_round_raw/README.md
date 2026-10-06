# Original four-round Qwen3-1.7B figures

These twelve PNG files are byte-for-byte copies of the supplied plotting
exports in `code/experiments/outputs/2026-10-03-h100`. Only the
`rounds0-4` exports are included, not the later eight-round views. Each
condition has two splits (ID/OOD) and two image types (aggregate three-panel
plot and difficulty-stratified accuracy). The conditions are unaudited,
adaptive-audit B=16, and their joint comparison.

The accompanying six `*_summary.json` files contain the exported means,
standard deviations, interval half-widths, and paired contrasts. The PNGs
were neither recolored nor redrawn; their original separate axis scales,
legends, and uncertainty bars are retained. No raw answers or adapter tensors
are added by this collection.

The appendix embeds all twelve images as six ID/OOD pairs in
`qwen_four_round_raw_figures.tex`. The restyled main-text Figure 3 is unchanged.
`provenance.json` records the exact source paths, SHA-256 digests, image sizes,
and source-summary checks. Its caption values agree with the independently
validated four-round manuscript records.

## Figure contract

- Question: How do accuracy, paired arm differences, and nonshortest-error
  incidence evolve over the fixed four-round Qwen horizon?
- Surface: The existing LaTeX appendix; original static PNG exports.
- Family: Discrete checkpoint comparison with intervals; small multiples by
  generator stratum. The five measured checkpoints are the full requested
  horizon, not a sampled subset of a longer trend.
- Grain: Five paired blocks per arm/policy/checkpoint, fixed 2,000 ID or 1,000
  OOD questions. Round zero is one common checkpoint with zero block variation.
- Colors: Original third-party source colors, R blue (`#1f77b4`), S red
  (`#d62728`), paired differences black; no new palette is introduced.
- Non-color encoding: R/S labels; solid/circle unaudited and dashed/square
  audited marks in the joint plots; explicit panels and zero reference lines.
- Footprint: Full appendix width, two original images per figure, ID above OOD.
- Supported reading: Both unaudited arms improve; endpoint accuracy contrasts
  include zero, while unaudited endpoint nonshortest-incidence contrasts have
  descriptive intervals above zero. Same-arm mean audit changes are negative
  but imprecise. No uniform audit benefit or stratum-specific significance is
  claimed.
- QA: Source-byte equality, summary statistics against saved evaluations,
  final LaTeX compilation, and rendered appendix-page inspection. The shared
  base is not treated as independent block replication. Intervals are
  descriptive and not multiplicity-adjusted.
