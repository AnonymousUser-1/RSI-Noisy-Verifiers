# Audit-threshold figures

Data and code of the audit-threshold figures (Figure 2): the parameter scan in `data/scan-v1/`, the final
figures in `figures/final/`, and the scripts that produce them.

```bash
python reproduce.py --out-dir NEW_DIRECTORY
```

regenerates the scan, checks that every CSV is byte-identical to `data/scan-v1/`, and redraws the three
figures. Dependencies are pinned in `requirements-lock.txt`.
