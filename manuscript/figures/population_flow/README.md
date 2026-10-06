# Population accuracy and stopped-gradient source panels

These source panels now form Figure 1(a,b) in the four-panel overview.
They are no longer included as a separate population-flow float.
It contains deterministic population-model calculations, not language-model
training measurements or evidence that the population theorem holds for a
different finite-budget auditing policy.

## Reproduce

From the parent directory containing `code/` and `manuscript/`:

```bash
python -m pip install -r code/requirements-figures.txt
python code/plot_population_flow.py --out manuscript/figures/population_flow --overwrite
```

The script imports the existing `code/rsi/logistic.py` and the independent
enumeration in `code/theory_checks.py`. It does not modify the model or invoke
the unrelated legacy `simulate.py` recursion. ReportLab exports a vector PDF;
`pdftoppm`, if available, additionally produces the PNG preview. DejaVu Sans
(matching Figure 1) is preferred, with Arial as a fallback. Fonts are embedded
when available, with Greek-coverage checks; otherwise
the standard PDF Helvetica/Symbol fonts are used. No GPU or model download is
required. Without `--overwrite`, existing named exports are refused.

## Settings and readouts

- Model: `a=0.75`, `b=0.5`, `L=10`, `theta0=0`.
- Trajectories: Euler integration over `t in [0,20]`, step `dt=0.0025`;
  8,001 points for each rule. Plot time is `t = step_index * dt`, not a count
  of neural training rounds.
- Gradient field: 601 evenly spaced points over `theta in [-0.25,1.25]`.
- At `t=20`: R accuracy `0.8774465360355461`; S accuracy
  `0.35958859256212267`. The finite-time R value is not the asymptotic limit.
- Negative S equilibrium: `theta*=-0.11146044018345069`, accuracy
  `0.35958859256212095`.
- Comparing `dt=0.005` with `0.0025` at common times gives maximum accuracy
  differences of `0.0001863654341072163` for R and
  `0.000208264191472185` for S. The next-coarser comparisons are approximately
  twice as large. These are refinement diagnostics, not rigorous bounds on
  the error relative to the exact continuous solution.
- Independent enumeration checks sampled trajectory gradients, accuracies,
  TPR/FPR, and all 1,202 rule/grid-point gradient values.
- Style matches Figure 1(b): R teal (`#127F83`)/solid; S muted plum
  (`#805677`)/dashed; thin neutral axes, sparse reference guides, and direct
  curve labels. Colors identify rules, not the sign of accuracy changes.
  No sampling error bars are drawn.
- The source PDF retains its compact two-panel layout. The composite overview
  is included at `0.7\textwidth`; source data, axis ranges, integration
  settings, and numerical annotations are preserved.

## Files

- `population_flow.pdf`: vector figure used by the manuscript.
- `population_flow.png`: 200-dpi preview rendered from that PDF.
- `trajectories.csv`: full R/S trajectories, including accuracy, gradients,
  acceptance rates, purity, and group-specific acceptance probabilities.
- `gradient_field.csv`: the parameter grid and both stopped gradients.
- `validation.json`: numerical checks, parameters, source/font hashes,
  environment versions, endpoint values, and export checksums.

The appendix documents numerical scope. Mathematical convergence is established
by the theorem and proof, not by a finite numerical plot.
