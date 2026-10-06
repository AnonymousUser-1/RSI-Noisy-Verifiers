# Validated graph additions

Sources: the H100 export updated 2026-10-04 23:35 UTC, restricted to rounds 0--4, and the complete 2026-10-04-one-step export.

From the manuscript directory:

```sh
python summarize_graph_extensions.py --code .. --out figures/graph_extensions
python verify_graph_extensions.py --code ..
python verify_h100_tables.py
```

Dependencies: Python 3, reportlab, pypdf. No GPU, model weights, network, or training.

`validated_results.json` stores the checked checkpoint/audit/diagnostic records and recomputed paired intervals; `table_contract.json` stores the rendered table-cell contract. `provenance.json` binds consumed source files and exports by SHA-256. The reference and held-out adapter tensors and raw held-out answers are not included: validation does not rejudge held-out responses or reconstruct gradients. Recorded original one-step acceptance reports passed 43/43 checks on the execution machines; adapter-hash rechecking needs their omitted adapters.

The one-step primary contrast is S-minus-R **Pass@1**, opposite in sign to the source export's S-minus-R error contrast. Intervals are descriptive pointwise block-t intervals (five blocks), unadjusted for multiple comparisons and conditional on fixed question sets. Each run retains its own base/null because platform-specific base predictions differ slightly.

Charts use the manuscript's Figure 1/2 teal/plum palette and vector rendering. R/S differ by line style, auditing by open/filled markers; one-step contrasts use labeled ID/OOD facets. Every requested round and block is retained, with no smoothing. QA uses the compiled manuscript pages.

The combined main-text endpoint table is also regenerated as `h100_endpoint_table.tex`, using separate five-block summaries for Llama and Qwen. Its audit contrasts are changes in **error** (negative favors auditing), not accuracy; the Qwen appendix audit-change table uses the opposite sign. Qwen base accuracy/error use four decimal places to avoid ambiguous rounding of 999/2000 and 1001/2000. The main-text Qwen accuracy figure has the same two-panel layout and 0.7-textwidth footprint as the Llama figure. The separate target-error plot remains in the appendix.
