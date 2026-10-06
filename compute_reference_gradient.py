#!/usr/bin/env python3
"""Compute the reference gradient `h = grad R_reference` for the matched experiment.

`h` is the vector the main diagnostic projects onto: `h^T Delta theta` (PLAN.md
section 4, HANDOFF.md deliverable B.3).  It is computed **once per common
initialisation point** -- the `shared_adapter/` the arms start from -- and shared
by that point's R, S and null arms.  It is not shared across different
initialisations: an h taken at one shared adapter is a gradient at that point only.

    python compute_reference_gradient.py --config CONFIG --data DATA \\
        --shared-adapter shared_adapter/ --out OUT [--seed 0]

Why this is a separate entry and not `rsi.reference_gradient`
-------------------------------------------------------------
`rsi/reference_gradient.py` is left as it is (blob `7c3b2bb0`).  Three of its
properties are not the ones this diagnostic needs:

  1. `normalize=True` by default (`:39`, `:159`): the default path returns a
     **unit vector**, the direction of `grad R_reference` and not the gradient.
     Here `normalize` is `False`, always; `--normalize` is refused, and the
     record says `normalize: false` beside the raw norm.
  2. `:87` multiplies HF's token-averaged batch loss by `input_ids.size(0)`, which
     weights each example by its share of the batch's supervised tokens, not by
     1/K.  With responses of unequal length the two differ.  Here each example's
     own response-token mean is taken (`response_losses`), then the K of them are
     averaged with weight 1/K each.
  3. It reads `batch['labels']` from a loader it does not build.  Here the encode
     is the shared one in `rsi.reference_encode`, the construction the training
     arms use.

Same functional as the arms
---------------------------
The per-example loop is `one_step.py::_run_hf_arm`'s: the same encode, the same
`response_losses(...).sum()` for loss_i, the same `torch.autograd.grad(loss_i / K)`,
the same `train()` mode with dropout 0 (`check_isolation`) and `use_cache` off, the
same seed call.  That fixes what `R_reference` *means* -- the reference risk.  It
does not by itself make `h` projectable: `h^T Delta theta` is defined by the LoRA
parameter names, shapes and order and by the initialisation they sit at, and those
are checked separately (below).

Label source
------------
`reference_answer(task)` -- the task's canonical answer, not a sampled completion.
This batch **selected** it (science-aux `58155b26`); it is not derived as the only
reading of the constraints: a frozen sampled answer would also be
sampling-independent and shareable, and would define a different reference risk.
A re-ruling flips `--label-source`.  The canonical answer's NLL is also not the
task risk over *all* legal answers: a graph task with several shortest paths has
one canonical path here.  This is a response-token NLL diagnostic on 64 held-out
prompts, used for diagnostics only -- never for selection or tuning (PLAN.md 5).

Projectable by check, not by assumption
---------------------------------------
After `h` is saved, a `GradientRecorder` is built on the same model from the saved
file -- exactly as the arms will load it -- and its `Delta theta` coordinates are
compared with `h`'s: names, order, shapes (`check_parameter_binding`).  The
initialisation identity is `verify_shared_adapter`'s parameter hash, base model and
revision, written into the record.

Output (OUT must not exist or must be empty; a written h is never overwritten)
  OUT/h.pt               torch.save({name: tensor}): what `GradientRecorder` loads.
  OUT/per_sample.jsonl   one line per reference prompt: id, label, token counts,
                         loss_i.
  OUT/h.json             written last: bindings (base model and revision, the
                         frozen base pin and its own hash, shared adapter and its
                         parameter hash, device, dtype, resolved config hash and
                         config file, data manifest / dataset /
                         gradient_reference hashes), code (git_commit, git_dirty,
                         source_hash), definitions, and the sha256 of h.pt and
                         per_sample.jsonl.  It never records its own hash.
"""
import argparse
import contextlib
import json
import math
from pathlib import Path

from rsi import cost
from rsi.base_pin import check_base_pin
from rsi.common import (digest, environment, file_hash, load_config, read_jsonl, require_explicit, verify_dataset,
                        write_json)
from rsi.provenance import git_state
from rsi.reference_encode import assert_supervised_last_token, check_parameter_binding, encode_example
from rsi.tasks import reference_answer

LABEL_SOURCES = ("reference_answer",)

# One place the normalization decision lives, so a caller cannot half-enforce it.
REQUIRED_NORMALIZE = False

LABEL_SOURCE_NOTE = ("selected for this batch (science-aux 58155b26), not derived as the only reading; "
                     "a re-ruling flips --label-source. Canonical-answer NLL, not the task risk over all "
                     "legal answers (graph tasks may have several shortest paths)")


def reference_rows(data):
    """The `gradient_reference` split: prompt-only rows, diagnostics only."""
    path = Path(data) / "gradient_reference.jsonl"
    if not path.exists():
        raise SystemExit("%s does not exist; generate_data.py writes it (PLAN.md section 5)" % path)
    rows = read_jsonl(path)
    if not rows:
        raise SystemExit("%s is empty" % path)
    return rows


def labels_for(rows, label_source):
    """`(row, label)` per prompt.  `reference_answer` is the only source this batch."""
    if label_source not in LABEL_SOURCES:
        raise SystemExit("Unknown label source %r; this batch allows %s"
                         % (label_source, ", ".join(LABEL_SOURCES)))
    return [(row, reference_answer(row)) for row in rows]


def lora_parameters(model):
    """The trainable LoRA parameters, in `named_parameters()` order.

    The same selection `GradientRecorder._get_lora_params` makes; the recorder
    round trip in `write_reference_gradient` checks that it still is.
    """
    return [(n, p) for n, p in model.named_parameters() if "lora" in n.lower() and p.requires_grad]


def compute_h(backend, pairs, seed):
    """`h = grad R_reference`: the mean over the K examples of grad loss_i.

    The loop is `one_step.py::_run_hf_arm`'s, minus the optimizer: loss_i is the
    example's own response-token mean, and its contribution is
    `autograd.grad(loss_i / K)`, taken on its own and summed.  The sum is kept in
    float32 on the CPU, which is what the recorder computes in (`.float()`).
    """
    import torch

    from rsi.backends import response_losses

    model = backend.model
    params = lora_parameters(model)
    if not params:
        raise SystemExit("No trainable LoRA parameters; load the shared adapter with trainable=True")
    model.train()  # as the arms; check_isolation has fixed dropout at 0, so this is deterministic
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    torch.manual_seed(seed)

    h = {name: torch.zeros(p.shape, dtype=torch.float32) for name, p in params}
    per_sample = []
    for row, label in pairs:
        input_ids, labels, prefix_length = encode_example(backend, row["prompt"], label)
        assert_supervised_last_token(labels, backend.tokenizer.eos_token_id, row["id"])
        ids = torch.tensor([input_ids], device=backend.device)
        target = torch.tensor([labels], device=backend.device)
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        loss = response_losses(logits, target).sum()  # loss_i, this example's response-token mean
        contribution = torch.autograd.grad(loss / len(pairs), [p for _, p in params])
        for (name, _), g in zip(params, contribution):
            h[name] += g.detach().float().cpu()
        per_sample.append({"task_id": row["id"], "label": label, "prefix_tokens": prefix_length,
                           "supervised_tokens": len(input_ids) - prefix_length,
                           "loss": float(loss.detach())})
    return h, per_sample


def write_reference_gradient(backend, pairs, out, seed, bindings, label_source):
    """Compute h, save it, prove the recorder reads it in its own coordinates, record.

    Needs the model alive (the recorder round trip reads its parameters), so it
    runs before the backend is closed.  Order of writes: h.pt, per_sample.jsonl,
    then h.json last, carrying the other two files' hashes.
    """
    import torch

    from rsi.gradient_recorder import GradientRecorder

    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("%s is not empty; a written reference gradient is never overwritten" % out)
    out.mkdir(parents=True, exist_ok=True)

    # The code that runs, read before it runs (as run_matched_experiment.py does).
    env, git = environment(), git_state()
    h, per_sample = compute_h(backend, pairs, seed)
    h_path = out / "h.pt"
    torch.save(h, h_path)

    # Load h exactly as an arm will, and compare it with the recorder's own
    # Delta theta coordinates.  `_dot_product` checks names only; this checks
    # names, order and shapes, so a mismatch is an error here and not a silent
    # wrong number in every arm's main diagnostic.
    recorder = GradientRecorder(backend.model, str(h_path))
    recorder.before_update()
    check_parameter_binding(recorder.h, {n: tuple(t.shape) for n, t in recorder.theta_before.items()},
                            "h.pt as GradientRecorder loads it")

    per_sample_path = out / "per_sample.jsonl"
    with open(per_sample_path, "w", newline="\n") as f:
        for entry in per_sample:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

    counts = [entry["supervised_tokens"] for entry in per_sample]
    record = {
        "h": {"file": h_path.name, "sha256": file_hash(h_path),
              "kind": "raw gradient of R_reference (not unit-normalised)",
              "l2_norm": math.sqrt(sum(float((t.double() ** 2).sum()) for t in h.values())),
              "parameter_names": list(h),
              "parameter_shapes": {n: list(t.shape) for n, t in h.items()},
              "dtype": "float32"},
        "per_sample": {"file": per_sample_path.name, "sha256": file_hash(per_sample_path)},
        "normalize": REQUIRED_NORMALIZE,
        "label_source": label_source,
        "label_source_note": LABEL_SOURCE_NOTE,
        "loss": {"per_example": "response-token mean NLL (rsi.backends.response_losses); "
                                "prompt masked, response + EOS supervised",
                 "aggregation": "mean over the K examples of the per-example mean (weight 1/K each)",
                 "token_weighted": False,
                 "contribution": "torch.autograd.grad(loss_i / K), as one_step.py::_run_hf_arm"},
        "examples": len(per_sample),
        "supervised_tokens": {"min": min(counts), "max": max(counts), "equal": len(set(counts)) == 1},
        "coordinates": "checked: GradientRecorder loaded h.pt; names, order and shapes equal its "
                       "Delta theta keys",
        "bindings": bindings,
        "code": {"git_commit": git["git_commit"], "git_dirty": git["git_dirty"], "git_changes": git["changes"],
                 "git_reason": git["reason"], "source_hash": env["source_hash"]},
        "seed": seed,
        "environment": env,
    }
    write_json(out / "h.json", record)
    return record


def main(args):
    if args.normalize != REQUIRED_NORMALIZE:
        raise SystemExit("normalize must be %r: a unit vector is not grad R_reference"
                         % REQUIRED_NORMALIZE)
    from one_step import check_isolation
    from rsi.experiment import pin_config

    config = load_config(args.config, args.seed, args.backend)
    if config["backend"] != "hf":
        raise SystemExit("The reference gradient needs a real model; the mock backend has no "
                         "trainable parameters that could carry one (KNOWN_GAPS.md forbids "
                         "substituting mock numbers for real ones)")
    # h always uses its independent reference split, never selected/audited data.
    # Allow an audit-enabled run config so its full config hash binds to the arms;
    # no online audit is performed while calculating this diagnostic reference.
    check_isolation(config, allow_auditing=True)
    # Pinned, as make_shared_adapter.py pins: verify_shared_adapter requires the
    # loaded revision to be a commit id, and the default "main" is not one.
    config = pin_config(config)
    # The resolved base is the one this study is pinned to, or nothing is taken.
    # Checked after the pin and before the dataset and any load, so the failure
    # is "wrong base" and never a half-taken h against a base we did not choose.
    # `h^T Delta theta` is compared across arms at one shared initialisation;
    # that initialisation is the study's only if the base is.
    try:
        base_pin = check_base_pin(config)
        require_explicit(args.config, "reference")  # every value it uses is in the config file
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # The dataset every other entry verifies; the reference split must be one of
    # the files its manifest binds, not a file that merely sits in the directory.
    dataset = verify_dataset(args.data)
    if "gradient_reference.jsonl" not in dataset["files"]:
        raise SystemExit("%s does not list gradient_reference.jsonl; h is taken only on the split "
                         "generate_data.py wrote and hashed" % (Path(args.data) / "manifest.json"))
    pairs = labels_for(reference_rows(args.data), args.label_source)

    from rsi.backends import backend as make_backend
    from rsi.determinism import enable_determinism
    from rsi.shared_adapter import lora_mismatch, verify_shared_adapter

    determinism = enable_determinism()  # as the arms, before the first GPU matrix product
    records = getattr(args, "cost_records", None)
    timing = (cost.stage(records, "reference", "shared", detail={"examples": len(pairs)})
              if records else contextlib.nullcontext())
    with timing as attempt:
        record = _take_h(args, config, base_pin, dataset, pairs, determinism, make_backend,
                         verify_shared_adapter, lora_mismatch)
        if attempt is not None:
            attempt.generated(0, 0)
    print("wrote %s (h: %d tensors, %d examples, |h| = %.6e)"
          % (args.out, len(record["h"]["parameter_names"]), record["examples"], record["h"]["l2_norm"]))
    return record


def _take_h(args, config, base_pin, dataset, pairs, determinism, make_backend, verify_shared_adapter, lora_mismatch):
    backend = make_backend(config, adapter=args.shared_adapter, trainable=True)
    try:
        try:
            shared = verify_shared_adapter(backend.model, args.shared_adapter,
                                           backend.model_name, backend.revision)
        except ValueError as exc:
            raise SystemExit("h must be taken at the arms' common initialisation: %s" % exc) from exc
        differ = lora_mismatch(shared, config["training"])
        if differ:
            raise SystemExit("The shared adapter %s was made with other LoRA settings than %s states: %s"
                             % (args.shared_adapter, args.config, "; ".join(differ)))
        # config_hash and data as run_matched_experiment.py binds them, so the two
        # records compare field by field.  config_hash covers the resolved config,
        # seed included (load_config writes --seed into it).
        bindings = {"base_model": backend.model_name, "base_revision": backend.revision,
                    "base_pin": base_pin,
                    "shared_adapter": str(args.shared_adapter),
                    "shared_adapter_parameter_hash": shared["parameter_hash"],
                    "device": backend.device, "dtype": config["dtype"], "determinism": determinism,
                    "config_hash": digest(config),
                    "config_file": {"path": str(Path(args.config).resolve()), "sha256": file_hash(args.config)},
                    "data": {"path": str(args.data), "manifest_sha256": file_hash(Path(args.data) / "manifest.json"),
                             "dataset_hash": digest(dataset),
                             "gradient_reference_sha256": file_hash(Path(args.data) / "gradient_reference.jsonl")}}
        return write_reference_gradient(backend, pairs, args.out, args.seed, bindings, args.label_source)
    finally:
        backend.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True, help="Dataset root containing gradient_reference.jsonl")
    p.add_argument("--shared-adapter", required=True, help="The shared_adapter/ the arms start from")
    p.add_argument("--out", required=True, help="Output directory (must not exist or be empty)")
    p.add_argument("--label-source", default="reference_answer", choices=LABEL_SOURCES,
                   help="Label of the reference risk; default reference_answer (this batch's selection)")
    p.add_argument("--normalize", action="store_true",
                   help="Refused: a unit vector is not grad R_reference")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["hf", "mock"])
    p.add_argument("--cost-records", help="write this attempt's cost record (rsi/cost.py) to this directory")
    main(p.parse_args())
