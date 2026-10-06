# Qwen3-4B graph: shared round-1 inputs

The inputs shared by every Qwen3-4B graph comparison in the paper (one-step, four-round unaudited and
four-round audited with B = 16):

| Folder or file | Content |
|---|---|
| `graph/` | The task data (seed 2027) with `manifest.json`; the same questions as `python generate_data.py --task graph --seed 2027` |
| `pools/` | Round-1 candidate pools `b00`-`b04`, 2,048 prompts x 8 answers each, with their `.meta.json`. `Qwen/Qwen3-4B` at `1cfa9a7208912126459214e8b04321603b3df60c`, temperature 1.3, top-p 1.0, top-k off, at most 2,048 new tokens, drawn without the repetition stop |
| `shared_adapter/` | The common LoRA initialisation (B = 0) |
| `matching/` | The round-1 matched R/S selections (K = 64, C:E = 48:16) and the K ladder |
| `reference_out_audit_legacy/`, `reference_out_audit_new/` | The reference gradient h for the two sampling profiles of the four-round runs (tensor-equal; `provenance/reference-equivalence.json`) |
| `configs/` | The configurations of both profiles, the evaluation configuration and the model pin |
| `provenance/` | The profile boundary (block b01, round 3), the pool verification and the reference equivalence |
| `PROTOCOL.json`, `PACKAGE_MANIFEST.json`, `SHA256SUMS.txt` | The protocol identity; size and SHA-256 of every file |

Verify the folder (CPU only):

```bash
python verify_inputs.py
```

Import it into the one-step experiment, from the repository root:

```bash
python experiments/import_pools.py graph_unaudited_qwen3-4b RSI-Qwen3-4B-Graph-paired-inputs-20261004
```

Paths inside the metadata are those of the machine that produced the files; they are kept as provenance.
