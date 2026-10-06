# Appendix TPR table: validation notes

Question: report round-one and later-round pre/post-audit true-positive acceptance rates for the completed Llama-3.2-3B graph comparison. The source is the 2026-10-03 16:06 UTC compact H100 export, not a newly executed experiment.

Assessment: share with the saved-summary caveat below. All 80 expected arm-rounds (two conditions × five blocks × two arms × four rounds) and 40 audit reports are present. Their original-pool counts, selection certificates, audit inputs/outputs, query counts, removal counts, and completion records are consistent. The independent display-table checker validates 914 numeric entries across the old and new tables, including the 80 entries added in Table 16. No earlier empirical values were changed.

## Definitions

- Pre-audit TPR: `48 / N_plus`.
- Post-audit retained TPR: `C_retained / N_plus`, including retained zero-weight rows.
- Post-audit positive-weight TPR: `C_positive / N_plus`, counting correct rows with weight strictly greater than zero, not their weight mass.
- `N_plus` is the same nontruncated original-pool correct count before and after an audit. It can differ across later arm-specific pools.
- The table displays `100 × mean(block rate)` over five blocks. It does not divide summed numerators by summed denominators. Denominator means and ranges are also displayed.
- Unaudited post-audit cells are not applicable and shown as dashes. No blocks or rounds are missing from this table.

All 40 audited arm-rounds retain 48 correct rows, so retained TPR equals pre-audit TPR. Positive-weight TPR can be lower. For example, audited b00/S/round 4 has `N_plus=9920`, 48 retained correct rows, and 37 positive-weight correct rows. Those ratios are independently spot-checked in the companion notebook.

## Inspectable artifacts

`block_rates.csv` contains all 80 block-level numerator/denominator records and relative source paths. `round_summary.csv` contains the 16 displayed condition/arm/round summaries. `validated_tpr.json` preserves both grains. `provenance.json` binds all 204 consumed JSON files and the new analysis script by SHA-256, separately from the earlier 395-file figure/result manifest.

From the manuscript directory, regenerate with:

```bash
python -B summarize_h100_tpr.py --snapshot ../experiments/outputs/2026-10-03-h100
python -B verify_h100_tables.py
```

The three code cells in `tpr_validation.ipynb` were executed top-to-bottom with a plain-Python cell executor, and recorded outputs were checked against that execution. A native Jupyter kernel was not used because `nbformat`, `nbclient`, and `ipykernel` are absent from this environment. In a Jupyter-enabled environment, the exact native check is:

```bash
python -m jupyter nbconvert --execute --to notebook --inplace tpr_validation.ipynb
```

## Caveats and scope

The export omits raw candidate responses and training weights. These checks establish saved count/rate consistency; they do not independently rejudge correctness, reconstruct weights, or certify the original GPU execution. The reported TPRs are finite-pool selection statistics, not held-out accuracy or population-verifier recall. Later selection TPR can fall when the correct pool grows despite a fixed quota of 48; that is not evidence that model accuracy declined. Incomplete model/task runs are outside this completed-pair table.

Only the appendix results source, the new table/data companions, and the display checker were edited for this addition. Existing main-text source was left unchanged. The previously compiled PDF did not fully reflect existing saved main-text edits, so its first eight extracted pages were not used as a byte-for-byte baseline for the rebuild.
